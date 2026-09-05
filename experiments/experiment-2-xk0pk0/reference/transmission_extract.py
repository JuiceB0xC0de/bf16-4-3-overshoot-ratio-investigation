"""Extract the four-stage transmission table from six-camera step bundles.

A training step is not one event.  It is a chain:

    optimizer intent -> numerical realization -> representation -> function

and nothing guarantees that consecutive stages resemble each other
geometrically.  Every chart, event card, and cross-step comparison wants the
same long-form table, so it gets built once here and never re-derived from
NPZ files downstream.

The load-bearing quantity this produces is the **overshoot index**.

For an ideal update ``i`` and a realized update ``r``, define

    delivery = |r| / |i|
    cosine   = <i, r> / (|i| |r|)

If realization were pure support truncation, keeping some coordinates exactly
and zeroing the rest, then ``r = P i`` for a projection ``P`` and

    cosine = <i, P i> / (|i| |P i|) = |P i|^2 / (|i| |P i|) = |P i| / |i| = delivery

so cosine and delivery are the *same number*, and their ratio is exactly 1.
Measured, that ratio sits near 0.75 and is stable across thousands of steps.
Realization is therefore not a mask.  The surviving coordinates are displaced
too, and

    overshoot = delivery / cosine = |r|^2 / <i, r>

is how much longer the realized update is than the part of it the intent
actually accounts for.  Round-to-nearest on the parameter's own bf16 lattice
predicts exactly this sign: an intended delta below half a ULP is flushed to
zero, and one just above half a ULP is snapped to a full ULP, which is larger
than intended.  Truncation loses magnitude; snapping overshoots it.

Everything here is read-only over local bundles.  No GPU, no network.

    python transmission_extract.py ~/second-pass/step_* -o transmission.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import orjson


# The only float32 parameters in this model.  Everything else is bfloat16, and
# that single fact explains the two-tier realization pattern that repeatedly
# gets reported as a property of parameter *families*.
FP32_SUFFIXES = (".mixer.D", ".mixer.dt_bias")

# Detailed per-coordinate families camera 3 retains in full.
DETAIL_FAMILIES = ("D", "angles", "dd_A", "dd_dt", "dt_bias", "trap")
FP32_FAMILIES = ("D", "dt_bias")

STAGES = {
    "grad_norm_preclip": ("intent", None),
    "clip_coefficient": ("intent", None),
    "clip_fired": ("intent", None),
    "ideal_norm": ("intent", None),
    "ideal_adaptive_norm": ("intent", None),
    "ideal_decay_norm": ("intent", None),
    "carry_norm": ("intent", None),
    "innovation_norm": ("intent", None),
    "innovation_over_carry": ("intent", None),
    "realized_norm": ("realization", "intent->realization"),
    "delivery": ("realization", "intent->realization"),
    "cosine": ("realization", "intent->realization"),
    "overshoot": ("realization", "intent->realization"),
    "zero_fraction": ("realization", "intent->realization"),
    "zero_fraction_bf16": ("realization", "intent->realization"),
    "zero_fraction_fp32": ("realization", "intent->realization"),
    "delivery_bf16": ("realization", "intent->realization"),
    "delivery_fp32": ("realization", "intent->realization"),
    "cosine_bf16": ("realization", "intent->realization"),
    "cosine_fp32": ("realization", "intent->realization"),
    "overshoot_bf16": ("realization", "intent->realization"),
    "embedding_delta_share": ("realization", "intent->realization"),
    "final_boundary_movement": ("representation", "realization->representation"),
    "peak_mixer_movement": ("representation", "realization->representation"),
    "peak_mixer_layer": ("representation", "realization->representation"),
    "mean_mixer_movement": ("representation", "realization->representation"),
    "loss_pre": ("function", "representation->function"),
    "loss_post": ("function", "representation->function"),
    "loss_change": ("function", "representation->function"),
    "loss_change_rel": ("function", "representation->function"),
}


class BundleCorrupt(Exception):
    """A file does not match the hash its own bundle recorded for it."""


def verify_dir(directory: Path) -> None:
    """Check every file in a camera directory against its own SHA256SUMS.

    Not optional.  A camera-4 capture was found sitting in a camera-6
    directory under the right filename; without this gate the extractor reads
    it, finds unexpected keys, and the failure looks like a schema problem
    rather than a corrupt bundle.
    """
    import hashlib

    sums = directory / "SHA256SUMS"
    if not sums.is_file():
        raise BundleCorrupt(f"{directory} has no SHA256SUMS")
    for line in sums.read_text().strip().splitlines():
        want, name = line.split("  ", 1)
        path = directory / name
        if not path.is_file():
            raise BundleCorrupt(f"{directory.name}/{name} is missing")
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != want:
            raise BundleCorrupt(f"{directory.name}/{name} hash mismatch")


def _npz(directory: Path, suffix: str):
    hits = sorted(directory.glob(f"*{suffix}"))
    if not hits:
        raise FileNotFoundError(f"no *{suffix} in {directory}")
    return np.load(hits[0])


# Manifest fields that must agree across every camera in one bundle.  If they
# disagree, the directory is a mixture of captures and nothing in it is safe to
# join on step number.
IDENTITY_FIELDS = (
    "run_id", "global_step", "local_step", "epoch", "row_index",
    "model_id", "model_revision", "grad_clip_norm", "lr",
)

# Carried through unchanged so a transmission row can be traced to the exact
# code and container that produced it.
PROVENANCE_FIELDS = ("image_ref", "camera_ids")


def read_provenance(step_dir: Path) -> dict:
    """Identity for one capture, cross-checked across all of its cameras.

    A transmission row with no ``run_id`` cannot be compared with a row from
    any other run, and this project has several runs with overlapping step
    numbers.  Provenance is read from the manifests rather than inferred from
    directory names, which are just labels.
    """
    manifests = {}
    for camera_dir in sorted(step_dir.glob("camera-*")):
        path = camera_dir / "manifest.json"
        if path.is_file():
            manifests[camera_dir.name] = json.loads(path.read_text())
    if not manifests:
        raise BundleCorrupt(f"{step_dir.name} has no camera manifests")

    identity, disagreements = {}, []
    for field in IDENTITY_FIELDS:
        values = {
            name: m[field] for name, m in manifests.items() if field in m
        }
        distinct = {json.dumps(v, sort_keys=True) for v in values.values()}
        if len(distinct) > 1:
            disagreements.append({"field": field, "values": values})
        identity[field] = next(iter(values.values()), None)
    if disagreements:
        raise BundleCorrupt(
            f"{step_dir.name}: cameras disagree on {[d['field'] for d in disagreements]}"
        )

    first = next(iter(manifests.values()))
    provenance = {key: first.get(key) for key in PROVENANCE_FIELDS}
    provenance.update(identity)
    provenance["cameras_present"] = sorted(manifests)
    provenance["finite_assertions_passed"] = all(
        m.get("finite_assertions_passed", False) for m in manifests.values()
    )
    provenance["source_bundle"] = step_dir.name

    run_id = identity.get("run_id")
    epoch = identity.get("epoch")
    local_step = identity.get("local_step")
    provenance["capture_id"] = (
        f"{run_id}:e{epoch}:s{local_step:06d}"
        if run_id and epoch is not None and local_step is not None
        else None
    )
    return provenance


def _loss(archive, *names: str) -> float:
    """Pull whichever loss key this phase actually used.

    The pre phase records ``training_loss`` from the real training forward; the
    post phase records ``post_update_loss`` from the no-grad replay.  They are
    different names for the same measurement at two ends of one step.
    """
    for name in names:
        if name in archive.files:
            return float(archive[name])
    raise KeyError(f"none of {names} in archive")


def _rel_l2(pre: np.ndarray, post: np.ndarray) -> float:
    pre64 = pre.astype(np.float64)
    post64 = post.astype(np.float64)
    denom = float(np.linalg.norm(pre64.ravel()))
    if denom == 0.0:
        return float("nan")
    return float(np.linalg.norm((post64 - pre64).ravel()) / denom)


def _pair(ideal_l2, realized_l2, inner, mask=None):
    """Pooled delivery / cosine / overshoot over a subset of tensors."""
    if mask is not None:
        ideal_l2, realized_l2, inner = ideal_l2[mask], realized_l2[mask], inner[mask]
    ideal = math.sqrt(float(np.sum(ideal_l2.astype(np.float64) ** 2)))
    realized = math.sqrt(float(np.sum(realized_l2.astype(np.float64) ** 2)))
    dot = float(np.sum(inner.astype(np.float64)))
    if ideal == 0.0 or realized == 0.0:
        return {"ideal": ideal, "realized": realized, "delivery": float("nan"),
                "cosine": float("nan"), "overshoot": float("nan")}
    delivery = realized / ideal
    cosine = dot / (ideal * realized)
    # overshoot == |r|^2 / <i,r>; 1.0 under pure support truncation.
    overshoot = delivery / cosine if cosine != 0.0 else float("nan")
    return {
        "ideal": ideal,
        "realized": realized,
        "delivery": delivery,
        "cosine": cosine,
        "overshoot": overshoot,
    }


def extract_camera3(path: Path) -> dict:
    z = _npz(path, "_optimizer_delivery.npz")

    names = [str(s) for s in z["parameter_names"]]
    numel = z["param_meta/numel"].astype(np.float64)
    ideal_l2 = z["param/ideal_total_l2"]
    delta_l2 = z["param/delta_l2"]
    inner = z["param/delivery_inner_product"]
    zero_frac = z["param/delta_fraction_zero"].astype(np.float64)

    is_fp32 = np.array([n.endswith(FP32_SUFFIXES) for n in names])
    is_embed = np.array([n == "backbone.embedding.weight" for n in names])

    whole = _pair(ideal_l2, delta_l2, inner)
    bf16 = _pair(ideal_l2, delta_l2, inner, ~is_fp32)
    fp32 = _pair(ideal_l2, delta_l2, inner, is_fp32)

    beta1 = float(z["param/group_beta1"][0])
    carry = beta1 * math.sqrt(float(np.sum(z["param/exp_avg_pre_l2"] ** 2)))
    innovation = (1.0 - beta1) * math.sqrt(
        float(np.sum(z["param/post_clip_grad_l2"] ** 2))
    )

    def weighted_zero(mask):
        if not mask.any():
            return float("nan")
        return float(np.sum(zero_frac[mask] * numel[mask]) / np.sum(numel[mask]))

    realized_sq = float(np.sum(delta_l2.astype(np.float64) ** 2))
    embed_share = (
        float(np.sum(delta_l2[is_embed].astype(np.float64) ** 2)) / realized_sq
        if realized_sq
        else float("nan")
    )

    # Per-tensor rows, so the two-tier pattern can be plotted rather than
    # asserted.  A tensor with no intended update contributes nothing.
    per_tensor = []
    for i, name in enumerate(names):
        if ideal_l2[i] == 0.0 or delta_l2[i] == 0.0:
            continue
        d = float(delta_l2[i]) / float(ideal_l2[i])
        c = float(inner[i]) / (float(ideal_l2[i]) * float(delta_l2[i]))
        per_tensor.append(
            {
                "name": name,
                "dtype": "float32" if is_fp32[i] else "bfloat16",
                "numel": int(numel[i]),
                "delivery": d,
                "cosine": c,
                "overshoot": d / c if c != 0.0 else float("nan"),
                "zero_fraction": float(zero_frac[i]),
            }
        )

    return {
        "dataset_row": int(z["dataset_row"]),
        "optimizer_step": int(z["optimizer_step"]),
        "training_loss": float(z["training_loss"]),
        "grad_norm_preclip": float(z["clip/observer_pre_clip_total_norm"]),
        "clip_coefficient": float(z["clip/clip_coefficient_empirical"]),
        "clip_fired": int(z["clip/clip_fired"]),
        "clip_max_norm": float(z["clip/clip_max_norm"]),
        "ideal_norm": whole["ideal"],
        "ideal_adaptive_norm": math.sqrt(
            float(np.sum(z["param/ideal_adaptive_l2"] ** 2))
        ),
        "ideal_decay_norm": math.sqrt(float(np.sum(z["param/ideal_decay_l2"] ** 2))),
        "carry_norm": carry,
        "innovation_norm": innovation,
        "innovation_over_carry": innovation / carry if carry else float("nan"),
        "realized_norm": whole["realized"],
        "delivery": whole["delivery"],
        "cosine": whole["cosine"],
        "overshoot": whole["overshoot"],
        "delivery_bf16": bf16["delivery"],
        "cosine_bf16": bf16["cosine"],
        "overshoot_bf16": bf16["overshoot"],
        "delivery_fp32": fp32["delivery"],
        "cosine_fp32": fp32["cosine"],
        "overshoot_fp32": fp32["overshoot"],
        "zero_fraction": weighted_zero(np.ones(len(names), dtype=bool)),
        "zero_fraction_bf16": weighted_zero(~is_fp32),
        "zero_fraction_fp32": weighted_zero(is_fp32),
        "embedding_delta_share": embed_share,
        "per_tensor": per_tensor,
    }


def extract_camera6(path: Path) -> dict:
    pre = _npz(path, "_pre_update.npz")
    post = _npz(path, "_post_update.npz")

    layers = sorted(
        {int(k.split("/")[0].split("_")[1]) for k in pre.files if k.startswith("layer_")}
    )
    per_layer = []
    for li in layers:
        row = {"layer": li}
        for quantity in ("mixer_output", "block_residual", "mixer_input"):
            key = f"layer_{li:02d}/{quantity}"
            if key in pre.files and key in post.files:
                row[quantity] = _rel_l2(pre[key], post[key])
        per_layer.append(row)

    mixer = [r["mixer_output"] for r in per_layer if "mixer_output" in r]
    peak_layer = (
        int(np.argmax(mixer)) if mixer and not all(math.isnan(v) for v in mixer) else -1
    )

    loss_pre = _loss(pre, "training_loss", "phase_loss", "post_update_loss")
    loss_post = _loss(post, "post_update_loss", "phase_loss", "training_loss")
    return {
        "final_boundary_movement": _rel_l2(
            pre["final_norm_output"], post["final_norm_output"]
        ),
        "peak_mixer_movement": float(np.max(mixer)) if mixer else float("nan"),
        "peak_mixer_layer": peak_layer,
        "mean_mixer_movement": float(np.mean(mixer)) if mixer else float("nan"),
        "loss_pre": loss_pre,
        "loss_post": loss_post,
        "loss_change": loss_post - loss_pre,
        "loss_change_rel": (loss_post - loss_pre) / loss_pre if loss_pre else float("nan"),
        "n_tokens": int(pre["supervised_token_count"]),
        "per_layer": per_layer,
    }


def extract_camera0(path: Path) -> dict:
    """Held-out probe, when the bundle has one.  Absent in pre-Camera-0 runs."""
    pre = _npz(path, "_pre_update.npz")
    post = _npz(path, "_post_update.npz")
    a = float(pre["probe_mean_nll_token_weighted"])
    b = float(post["probe_mean_nll_token_weighted"])
    return {
        "probe_pre": a,
        "probe_post": b,
        "probe_change": b - a,
        "probe_change_rel": (b - a) / a if a else float("nan"),
    }


def extract_step(step_dir: Path) -> dict | None:
    c3, c6 = step_dir / "camera-3", step_dir / "camera-6"
    if not c3.is_dir() or not c6.is_dir():
        return None

    verify_dir(c3)
    verify_dir(c6)
    record = {"step_dir": step_dir.name}
    record.update(extract_camera3(c3))
    record.update(extract_camera6(c6))

    c0 = step_dir / "camera-0"
    record["has_probe"] = c0.is_dir()
    if record["has_probe"]:
        verify_dir(c0)
        record.update(extract_camera0(c0))

    # The efficacy axis.  Same-batch loss is contaminated by example difficulty
    # (per-row loss correlates 0.893 across epochs), so the probe is preferred
    # wherever it exists and the choice is recorded rather than assumed.
    if record["has_probe"]:
        record["efficacy"] = -record["probe_change_rel"]
        record["efficacy_source"] = "heldout_probe"
    else:
        record["efficacy"] = -record["loss_change_rel"]
        record["efficacy_source"] = "same_batch_CONTAMINATED"
    return record


def long_form(records: list[dict]) -> list[dict]:
    rows = []
    for rec in records:
        for metric, (stage, transition) in STAGES.items():
            if metric not in rec:
                continue
            value = rec[metric]
            if isinstance(value, float) and math.isnan(value):
                continue
            rows.append(
                {
                    "step": rec["optimizer_step"],
                    "stage": stage,
                    "transition": transition,
                    "metric": metric,
                    "value": float(value),
                    "layer": None,
                    "family": None,
                }
            )
        for layer_row in rec["per_layer"]:
            for quantity, value in layer_row.items():
                if quantity == "layer" or math.isnan(value):
                    continue
                rows.append(
                    {
                        "step": rec["optimizer_step"],
                        "stage": "representation",
                        "transition": "realization->representation",
                        "metric": f"movement_{quantity}",
                        "value": float(value),
                        "layer": layer_row["layer"],
                        "family": None,
                    }
                )
        for tensor in rec["per_tensor"]:
            for metric in ("delivery", "cosine", "overshoot", "zero_fraction"):
                value = tensor[metric]
                if math.isnan(value):
                    continue
                rows.append(
                    {
                        "step": rec["optimizer_step"],
                        "stage": "realization",
                        "transition": "intent->realization",
                        "metric": f"tensor_{metric}",
                        "value": float(value),
                        "layer": None,
                        "family": tensor["dtype"],
                        "tensor": tensor["name"],
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundles", nargs="+", type=Path)
    parser.add_argument("-o", "--out", type=Path, default=Path("transmission.json"))
    args = parser.parse_args()

    records = []
    for directory in sorted(args.bundles):
        if not directory.is_dir():
            continue
        try:
            record = extract_step(directory)
        except Exception as error:
            print(f"  {directory.name}: SKIPPED ({error})")
            continue
        if record is None:
            print(f"  {directory.name}: SKIPPED (no camera-3/camera-6)")
            continue
        records.append(record)
        print(
            f"  {directory.name}: step {record['optimizer_step']:>6} "
            f"row {record['dataset_row']:>5}  "
            f"grad {record['grad_norm_preclip']:6.3f} "
            f"clip {record['clip_coefficient']:.3f}  "
            f"deliv {record['delivery']:.4f} cos {record['cosine']:.4f} "
            f"overshoot {record['overshoot']:.4f}  "
            f"loss {record['loss_change_rel'] * 100:+.3f}%"
        )

    if not records:
        raise SystemExit("no bundles extracted")

    records.sort(key=lambda r: r["optimizer_step"])
    payload = {
        "schema": "mamba3-transmission/1",
        "n_steps": len(records),
        "any_probe": any(r["has_probe"] for r in records),
        "steps": records,
        "long_form": long_form(records),
    }
    args.out.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))

    print(f"\nwrote {args.out}  ({len(records)} steps, "
          f"{len(payload['long_form'])} long-form rows)")
    if not payload["any_probe"]:
        print("NOTE: no camera-0 in any bundle, so the efficacy axis is "
              "same-batch loss and carries the example-difficulty confound.")

    over = [r["overshoot"] for r in records if not math.isnan(r["overshoot"])]
    if over:
        print(f"\novershoot index across {len(over)} steps: "
              f"min {min(over):.4f}  median {sorted(over)[len(over) // 2]:.4f}  "
              f"max {max(over):.4f}")
        print("1.0 would mean realization is pure support truncation.")


if __name__ == "__main__":
    main()
