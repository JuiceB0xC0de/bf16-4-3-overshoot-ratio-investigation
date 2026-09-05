"""Shadow optimizers: the overshoot-versus-scale curve on one common trajectory.

Why this exists
---------------
The six planned runs let each optimizer walk its own trajectory, so a
difference between them confounds "this optimizer's update distribution" with
"this optimizer's model state after N different steps".  The 20-step SGD sweep
made that confound expensive to ignore twice over:

* SGD's pooled overshoot moves monotonically with learning rate -- 1.3186,
  1.2594, 1.2146, 1.1828 at lr 0.002, 0.005, 0.01, 0.02.  The number is not a
  constant of the optimizer, so a single matched operating point is a single
  sample from a curve, and which point you pick decides the answer.
* SGD's scale-matched learning rate (0.0583) is one that destabilizes
  training, so its trajectory is not comparable to AdamW's healthy one.

The shadow fixes both.  Every optimizer's update is evaluated against the
*same* model state, the *same* post-clip gradients, and the *same* bfloat16
lattice, over a grid of intended-update magnitudes.  Nothing varies but the
rule that turns a gradient into a direction.

    intent = alpha * u_optimizer      with alpha chosen so ||intent|| = target
    realized = bf16(p_pre + intent) - p_pre

The directions ``u`` are learning-rate independent (a momentum buffer
accumulates gradients, not steps), so one pass computes the whole curve.

Weight decay is omitted from the shadow.  At AdamW's learning rate that is
exact rather than an approximation: the AdamW run measures
``decay_x_max = 0.00051``, so the decay term is ~1000x below the half-ULP
rounding threshold at every coordinate and cannot change a single rounding
decision.  It is NOT exact at the much larger learning rates the SGD
conditions need, where ``decay_x_max`` reaches 0.0967 and 0.1487 -- large
enough to move roundings.  The shadow's SGD rule is therefore a decay-free
SGD, and its numbers must be read as such; the real SGD runs, which carry the
decay, are reported separately.

Self-check
----------
``self_check`` below always compares ``shadow_adamw`` at the host's own
operating point against the host's measured overshoot.  That is a genuine
reconstruction test only on the AdamW host, where it maintains its own moment
buffers with its own recursion and its own model of the bf16 write, so
agreement tests the momentum recursions, the direction algebra, and the write
model at once.  On the other hosts it compares mismatched rules and its
``ok`` flag is meaningless -- the per-host reconstruction test is instead the
host's *own* rule read off the shadow curve at
``REFERENCE_MULTIPLIER``, which matches the host's intended-update norm to six
figures.  That reads within 0.003978 on the AdamW and Muon hosts and 0.054 to
0.119 low on the SGD hosts, the omitted decay.
"""

from __future__ import annotations

import math

import torch

from conditions import BETAS, EPS, MUON_MOMENTUM, SGD_MOMENTUM
from geometry import LatticeAccumulator, PooledAccumulator
from vendor_muon import muon_update

# Intended-update magnitudes, as multiples of AdamW's own pooled intent norm at
# that step.  Fixed here rather than chosen after seeing results.
SCALE_MULTIPLIERS = (1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0, 2.0, 4.0, 8.0)
REFERENCE_MULTIPLIER = 1.0


def _bf16_write(p_pre: torch.Tensor, intent: torch.Tensor) -> torch.Tensor:
    """The realized move when ``intent`` is added to a bfloat16 parameter.

    Mirrors the real path: the addition happens in float32 (the precision the
    observed pipeline used) and the result is stored back through bfloat16.
    """
    updated = (p_pre + intent).to(torch.bfloat16).to(torch.float32)
    return updated - p_pre


class ShadowOptimizers:
    """SGD, Muon and AdamW directions maintained on one gradient sequence."""

    def __init__(self, records: list):
        self.records = [r for r in records if r.muon_covered]
        self.state = {}
        self.step_count = 0

    @torch.no_grad()
    def _direction(self, name: str, record, grad: torch.Tensor) -> torch.Tensor:
        """The optimizer's update direction, before any learning rate."""
        key = (name, record.index)
        st = self.state.setdefault(key, {})

        if name == "sgd":
            if "buf" not in st:
                st["buf"] = grad.clone()
            else:
                st["buf"].mul_(SGD_MOMENTUM).add_(grad)
            return st["buf"].to(torch.float32)

        if name == "muon":
            if "buf" not in st:
                st["buf"] = torch.zeros_like(grad)
            # muon_update mutates the gradient it is given, so it gets a copy.
            update = muon_update(grad.clone(), st["buf"], beta=MUON_MOMENTUM,
                                 ns_steps=5, nesterov=True)
            return update.reshape(record.param.shape).to(torch.float32)

        if name == "adamw":
            beta1, beta2 = BETAS
            if "m" not in st:
                st["m"] = torch.zeros_like(grad)
                st["v"] = torch.zeros_like(grad)
            st["m"].mul_(beta1).add_(grad, alpha=1 - beta1)
            st["v"].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            t = self.step_count
            bc1 = 1.0 - beta1 ** t
            bc2 = 1.0 - beta2 ** t
            m = st["m"].to(torch.float32)
            v = st["v"].to(torch.float32)
            return (m / bc1) / (v.sqrt() / math.sqrt(bc2) + EPS)

        raise ValueError(name)

    @torch.no_grad()
    def measure(self, pre_params: dict, grads: dict, adamw_intent_norm: float,
                want_geometry: bool = False) -> dict:
        """One step of the shadow: every optimizer, every scale multiplier.

        ``adamw_intent_norm`` is the real AdamW run's pooled intent norm over
        this same coordinate set at this step, so multiplier 1.0 is AdamW's own
        operating point and every curve is read against it.
        """
        self.step_count += 1
        out = {}
        for name in ("adamw", "sgd", "muon"):
            directions = {}
            norm_sq = 0.0
            for r in self.records:
                u = self._direction(name, r, grads[r.index])
                directions[r.index] = u
                norm_sq += float(torch.sum(u * u, dtype=torch.float64))
            u_norm = math.sqrt(norm_sq)
            if u_norm == 0.0:
                continue

            per_scale = {}
            for mult in SCALE_MULTIPLIERS:
                target = adamw_intent_norm * mult
                alpha = target / u_norm
                pooled = PooledAccumulator()
                lattice = (LatticeAccumulator(self.records[0].param.device)
                           if want_geometry and mult == REFERENCE_MULTIPLIER
                           else None)
                for r in self.records:
                    p_pre = pre_params[r.index].to(torch.float32)
                    intent = directions[r.index] * -alpha
                    realized = _bf16_write(p_pre, intent)
                    pooled.add(intent, realized)
                    if lattice is not None:
                        lattice.add(p_pre, intent, realized)
                entry = {"target_intent_norm": target, "multiplier": mult,
                         **pooled.result()}
                if lattice is not None:
                    entry["geometry"] = lattice.result()
                per_scale[f"{mult:.6g}"] = entry
            out[name] = per_scale
            del directions
        return out


def self_check(shadow_row: dict, measured_overshoot: float,
               tol: float = 0.01) -> dict:
    """Shadow AdamW at multiplier 1 must reproduce the real AdamW step.

    The shadow keeps its own moment buffers, applies its own recursion, omits
    weight decay, and models the bf16 store itself.  Reproducing the number the
    real ``torch.optim.AdamW`` step produced therefore tests all of that at
    once -- including the claim that decay is below the rounding threshold.

    This is only meaningful on an AdamW host.  It is always called, so the
    eight non-AdamW rows record ``ok: False`` for comparing shadow AdamW
    against a different host rule; read those rows' reconstruction from the
    shadow curve for the host's own rule instead.
    """
    got = shadow_row["adamw"][f"{REFERENCE_MULTIPLIER:.6g}"]["overshoot"]
    return {
        "shadow_adamw_overshoot": got,
        "real_adamw_overshoot": measured_overshoot,
        "abs_gap": abs(got - measured_overshoot),
        "tol": tol,
        "ok": abs(got - measured_overshoot) <= tol,
    }
