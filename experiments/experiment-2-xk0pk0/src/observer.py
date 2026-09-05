"""Read-only per-step capture of intent, realization, and lattice geometry.

Structurally passive, like the project's camera 3: no module hooks, no tensor
hooks, no call to ``backward``, ``clip_grad_norm_`` or ``optimizer.step()``.
The driver owns all three; this class only takes detached clones either side of
the step and subtracts.

    delta = float32(p_post) - float32(p_pre)          # exactly, as recorded
    intent = the condition's own fp32 update rule from post-step native state
    delivery = |r|/|i| ; cosine = <i,r>/(|i||r|) ; overshoot = delivery/cosine

Three coordinate subsets are pooled independently, using the same rule as
``transmission_extract.py::_pair``:

* ``whole``  -- every optimizer parameter, the published headline number
* ``bf16``   -- the 122 natively-bfloat16 parameters
* ``muon_covered`` -- the 2-D hidden weights Muon actually orthogonalizes,
  which is the primary comparison set because it is the only one where all
  three conditions have the same optimizer writing the same coordinates

Lattice geometry is accumulated over ``bf16`` and ``muon_covered``.  The full
realized step-size distribution in ULPs is kept, not a zero/nonzero split:
Arm A found 35.88% of nonzero moves are two or more ULPs, so a two-bucket
summary would hide exactly where an optimizer difference should appear.
"""

from __future__ import annotations

import math

import torch

from geometry import LatticeAccumulator, PooledAccumulator, assert_on_lattice

SUBSETS = ("whole", "bf16", "muon_covered")
GEOMETRY_SUBSETS = ("bf16", "muon_covered")


class DeliveryObserver:
    def __init__(self, records: list, condition, device):
        self.records = records
        self.condition = condition
        self.device = device
        self._pre = {}
        self._lattice_checked = False

    @torch.no_grad()
    def snapshot(self) -> None:
        """Native-dtype clone of every parameter, before the trainer's step."""
        self._pre = {r.index: r.param.detach().clone() for r in self.records}

    @torch.no_grad()
    def capture(self, want_geometry: bool = True, want_per_tensor: bool = False) -> dict:
        if not self._pre:
            raise RuntimeError("capture() called without a matching snapshot()")

        pooled = {name: PooledAccumulator() for name in SUBSETS}
        lattice = ({name: LatticeAccumulator(self.device) for name in GEOMETRY_SUBSETS}
                   if want_geometry else {})
        per_tensor = []

        for r in self.records:
            pre_native = self._pre.pop(r.index)
            p_pre32 = pre_native.to(torch.float32)
            p_post32 = r.param.detach().to(torch.float32)
            delta = p_post32 - p_pre32
            intent, decay = self.condition.intent(r, p_pre32)

            if not self._lattice_checked and not r.native_fp32:
                assert_on_lattice(p_pre32)

            pooled["whole"].add(intent, delta)
            key = "fp32" if r.native_fp32 else "bf16"
            if key == "bf16":
                pooled["bf16"].add(intent, delta)
                if want_geometry:
                    lattice["bf16"].add(p_pre32, intent, delta, decay)
            if r.muon_covered:
                pooled["muon_covered"].add(intent, delta)
                if want_geometry:
                    lattice["muon_covered"].add(p_pre32, intent, delta, decay)

            if want_per_tensor:
                i_l2 = float(torch.linalg.vector_norm(intent, dtype=torch.float64))
                d_l2 = float(torch.linalg.vector_norm(delta, dtype=torch.float64))
                inner = float(torch.sum(intent * delta, dtype=torch.float64))
                if i_l2 > 0 and d_l2 > 0:
                    delivery = d_l2 / i_l2
                    cosine = inner / (i_l2 * d_l2)
                    per_tensor.append({
                        "name": r.name,
                        "dtype": "float32" if r.native_fp32 else "bfloat16",
                        "numel": r.numel,
                        "ndim": r.ndim,
                        "muon_covered": r.muon_covered,
                        "updater": r.updater,
                        "delivery": delivery,
                        "cosine": cosine,
                        "overshoot": delivery / cosine if cosine else float("nan"),
                        "zero_fraction": float((delta == 0).to(torch.float32).mean()),
                    })
            del pre_native, p_pre32, p_post32, delta, intent, decay

        self._lattice_checked = True
        self._pre = {}
        out = {"pooled": {k: v.result() for k, v in pooled.items()}}
        if want_geometry:
            out["geometry"] = {k: v.result() for k, v in lattice.items()}
        if want_per_tensor:
            out["per_tensor"] = per_tensor
        return out


@torch.no_grad()
def grad_stats(records: list, max_norm: float) -> dict:
    """Global gradient norm and the clip coefficient the trainer will apply."""
    total = math.sqrt(sum(
        float(torch.sum(r.param.grad.to(torch.float32) ** 2, dtype=torch.float64))
        for r in records if r.param.grad is not None))
    return {
        "grad_norm_preclip_observer": total,
        "clip_coefficient_expected": min(1.0, max_norm / (total + 1e-6)),
    }


def scalar_row(step: int, row_index: int, loss: float, capture: dict,
               clip: dict) -> dict:
    """The per-step record, flat enough to go straight into a table or W&B."""
    row = {"step": step, "dataset_row": row_index, "loss": loss}
    row.update(clip)
    for subset, stats in capture["pooled"].items():
        for key in ("ideal", "realized", "delivery", "cosine", "overshoot",
                    "zero_fraction", "residual_norm"):
            row[f"{subset}/{key}"] = stats[key]
    for subset, geo in capture.get("geometry", {}).items():
        for key in ("zero_step_fraction", "abs_step_ge_2_share_of_moved",
                    "mean_abs_step_of_moved", "ratio_predicted_from_x",
                    "ratio_from_realized_steps", "x_mean", "decay_x_max",
                    "multi_step_share_of_numerator"):
            row[f"geom_{subset}/{key}"] = geo[key]
        row[f"geom_{subset}/x_median"] = geo["x_quantiles_from_hist"]["0.5"]
    return row
