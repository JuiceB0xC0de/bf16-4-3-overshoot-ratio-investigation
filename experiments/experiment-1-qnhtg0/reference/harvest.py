"""Harvest transmission rows straight from the HuggingFace dataset.

The bundles are already published, so nothing here touches a Modal Volume and
nothing downloads a full capture.  An NPZ is a zip archive: given a seekable
file object, numpy reads the central directory and then only the members it is
asked for.  ``HfFileSystem`` provides that seekable object over HTTP, so a
capture costs a few hundred kilobytes instead of ~30 MB.

Every row carries its own provenance.  Runs in this project reuse step
numbers, so a transmission row without a ``run_id`` cannot be compared with
anything, and merging two runs on step number alone silently invents results.
``run_id``, ``capture_id``, epoch, local step, global step and dataset row are
read from the camera manifests rather than parsed out of directory names.

Output is one JSON object per line, one line per capture.

**The clip-8 and clip-1 runs are not a controlled A/B.**  They share a dataset
order and a model revision, and each capture pairs to its twin by
``pair_id = epoch{e}:row{r}``, but the two trajectories separate from the
first optimizer step: clip 1 fired on 100.00% of steps at mean scale 0.278, so
by step 100 the clip-1 model has taken 99 differently scaled updates.  At step
100 on the identical dataset row 99 the pre-clip gradient norms are 3.3875 and
2.9887 respectively, which is direct evidence the models differ there.

So a difference between paired captures at step N confounds "what clipping
does to this step" with "N steps of accumulated divergence."  What survives
without that confound is any quantity that is scale-free and computed entirely
within one step, such as ``overshoot``: if it is unchanged across a ~3.6x
difference in intended update magnitude, that is an invariance of the
numerical filter and does not require the two models to match.

    python harvest.py --limit 2
    python harvest.py --record clip8 -o capture_summary_clip8.jsonl
    python harvest.py --record clip1 -o capture_summary_clip1.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np


REPO = "datasets/juiceb0xc0de/mmb3-slt-6xcam-results"

# Published records.  The clip-1 run predates the ``grad_clip_norm`` field in
# run_summary.json, so the clip ceiling is taken from camera 3's own
# ``clip/clip_max_norm`` per capture rather than from run config.
RECORDS = {
    "clip8": "observed-runs/second-pass_clip8_148-observations",
    "clip1": "observed-runs/20260810T091311Z",
}
RECORD = RECORDS["clip8"]

FP32_SUFFIXES = (".mixer.D", ".mixer.dt_bias")

# Only these camera-3 members are read.  Everything else in the archive stays
# on the server.
C3_KEYS = (
    "param/ideal_total_l2", "param/delta_l2", "param/delivery_inner_product",
    "param/delta_fraction_zero", "param/ideal_adaptive_l2",
    "param/ideal_decay_l2", "param/exp_avg_pre_l2", "param/post_clip_grad_l2",
    "param/pre_clip_grad_l2", "param/group_beta1", "param_meta/numel",
    "parameter_names", "dataset_row", "optimizer_step", "training_loss",
    "clip/observer_pre_clip_total_norm", "clip/clip_coefficient_empirical",
    "clip/clip_fired", "clip/clip_max_norm",
)


def norm(values) -> float:
    return math.sqrt(float(np.sum(np.asarray(values, dtype=np.float64) ** 2)))


def pooled(ideal_l2, delta_l2, inner, mask=None) -> dict:
    """delivery / cosine / overshoot over a subset of tensors.

    overshoot == delivery / cosine == |r|^2 / <i,r>.  It is exactly 1.0 when
    realization is pure support truncation, because then r = P i and cosine
    and delivery are algebraically the same number.
    """
    if mask is not None:
        ideal_l2, delta_l2, inner = ideal_l2[mask], delta_l2[mask], inner[mask]
    i, r = norm(ideal_l2), norm(delta_l2)
    dot = float(np.sum(np.asarray(inner, dtype=np.float64)))
    if i == 0.0 or r == 0.0:
        return {"delivery": None, "cosine": None, "overshoot": None,
                "ideal_norm": i, "realized_norm": r}
    d, c = r / i, dot / (i * r)
    return {
        "ideal_norm": i,
        "realized_norm": r,
        "delivery": d,
        "cosine": c,
        "overshoot": d / c if c else None,
    }


def harvest_capture(fs, capture: str, meta: dict, with_boundary: bool,
                    record: str = RECORD) -> dict:
    base = f"{REPO}/{record}/{capture}"

    manifest = json.loads(fs.cat_file(f"{base}/camera-3/manifest.json"))

    with fs.open(
        next(p for p in fs.ls(f"{base}/camera-3", detail=False) if p.endswith(".npz")),
        "rb",
    ) as handle:
        z = np.load(handle)
        got = {key: z[key] for key in C3_KEYS if key in z.files}

    names = [str(s) for s in got["parameter_names"]]
    is_fp32 = np.array([n.endswith(FP32_SUFFIXES) for n in names])
    is_embed = np.array([n == "backbone.embedding.weight" for n in names])
    ideal_l2 = got["param/ideal_total_l2"]
    delta_l2 = got["param/delta_l2"]
    inner = got["param/delivery_inner_product"]
    numel = got["param_meta/numel"].astype(np.float64)
    zero = got["param/delta_fraction_zero"].astype(np.float64)

    whole = pooled(ideal_l2, delta_l2, inner)
    bf16 = pooled(ideal_l2, delta_l2, inner, ~is_fp32)
    fp32 = pooled(ideal_l2, delta_l2, inner, is_fp32)

    def zero_frac(mask):
        if not mask.any():
            return None
        return float(np.sum(zero[mask] * numel[mask]) / np.sum(numel[mask]))

    beta1 = float(got["param/group_beta1"][0])
    carry = beta1 * norm(got["param/exp_avg_pre_l2"])
    innovation = (1.0 - beta1) * norm(got["param/post_clip_grad_l2"])
    realized_sq = float(np.sum(delta_l2.astype(np.float64) ** 2))

    row = {
        # provenance, read from the manifest, never parsed from a folder name
        "run_id": manifest.get("run_id"),
        "capture_id": (
            f"{manifest.get('run_id')}:e{manifest.get('epoch')}"
            f":s{int(manifest.get('local_step', 0)):06d}"
        ),
        "epoch": manifest.get("epoch"),
        "local_step": manifest.get("local_step"),
        "global_step": manifest.get("global_step"),
        "dataset_row": int(got["dataset_row"]),
        "capture_type": meta.get("capture_type", "unknown"),
        # Pairing key for the clip-8 / clip-1 comparison.  Epoch plus dataset
        # row, not step number: it identifies the same example at the same
        # optimizer age in either run.  It does NOT assert the two runs hold
        # the same model at that point, and they do not; see the divergence
        # note in the module docstring.
        "pair_id": f"epoch{manifest.get('epoch')}:row{int(got['dataset_row'])}",
        "max_grad_norm": float(got["clip/clip_max_norm"]),
        "condition": f"clip{float(got['clip/clip_max_norm']):g}",
        "source_bundle": f"{record}/{capture}",
        "source_repo": REPO,
        "model_id": manifest.get("model_id"),
        "model_revision": manifest.get("model_revision"),
        "image_ref": manifest.get("image_ref"),
        "grad_clip_norm": manifest.get("grad_clip_norm"),
        "lr": manifest.get("lr"),
        "cameras_present": manifest.get("camera_ids"),
        "finite_assertions_passed": manifest.get("finite_assertions_passed"),
        # stage 1, optimizer intent
        "grad_norm_preclip": float(got["clip/observer_pre_clip_total_norm"]),
        "clip_coefficient": float(got["clip/clip_coefficient_empirical"]),
        "clip_fired": int(got["clip/clip_fired"]),
        "ideal_norm": whole["ideal_norm"],
        "ideal_adaptive_norm": norm(got["param/ideal_adaptive_l2"]),
        "ideal_decay_norm": norm(got["param/ideal_decay_l2"]),
        "carry_norm": carry,
        "innovation_norm": innovation,
        "innovation_over_carry": innovation / carry if carry else None,
        # transition 1, intent -> realization
        "realized_norm": whole["realized_norm"],
        "delivery": whole["delivery"],
        "cosine": whole["cosine"],
        "overshoot": whole["overshoot"],
        "delivery_bf16": bf16["delivery"],
        "cosine_bf16": bf16["cosine"],
        "overshoot_bf16": bf16["overshoot"],
        "delivery_fp32": fp32["delivery"],
        "cosine_fp32": fp32["cosine"],
        "overshoot_fp32": fp32["overshoot"],
        "zero_fraction": zero_frac(np.ones(len(names), dtype=bool)),
        "zero_fraction_bf16": zero_frac(~is_fp32),
        "zero_fraction_fp32": zero_frac(is_fp32),
        "embedding_delta_share": (
            float(np.sum(delta_l2[is_embed].astype(np.float64) ** 2)) / realized_sq
            if realized_sq else None
        ),
        # stage 4, function.  Same-batch only: this run predates camera 0, so
        # the efficacy axis carries the example-difficulty confound.
        "loss_pre": float(got["training_loss"]),
        "loss_post": None,
        "efficacy_source": "same_batch_CONTAMINATED",
        "has_probe": False,
        # validity
        "valid_realization": whole["delivery"] is not None,
        "valid_representation": False,
    }

    if with_boundary:
        c6 = f"{base}/camera-6"
        files = sorted(p for p in fs.ls(c6, detail=False) if p.endswith(".npz"))
        pre_p = next(p for p in files if "pre_update" in p)
        post_p = next(p for p in files if "post_update" in p)
        with fs.open(pre_p, "rb") as h:
            zp = np.load(h)
            pre = zp["final_norm_output"].astype(np.float64)
            row["loss_pre"] = float(zp["training_loss"])
        with fs.open(post_p, "rb") as h:
            zq = np.load(h)
            post = zq["final_norm_output"].astype(np.float64)
            key = "post_update_loss" if "post_update_loss" in zq.files else "phase_loss"
            row["loss_post"] = float(zq[key])
        denom = float(np.linalg.norm(pre.ravel()))
        row["final_boundary_movement"] = (
            float(np.linalg.norm((post - pre).ravel()) / denom) if denom else None
        )
        row["valid_representation"] = row["final_boundary_movement"] is not None

    if row["loss_post"] is not None and row["loss_pre"]:
        row["loss_change"] = row["loss_post"] - row["loss_pre"]
        row["loss_change_rel"] = row["loss_change"] / row["loss_pre"]
        row["efficacy"] = -row["loss_change_rel"]
    else:
        row["loss_change"] = row["loss_change_rel"] = row["efficacy"] = None
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out", type=Path, default=Path("capture_summary.jsonl"))
    parser.add_argument("--record", default="clip8",
                        help=f"one of {sorted(RECORDS)} or a full repo path")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8,
                        help="parallel captures; this job is network-bound")
    parser.add_argument("--no-boundary", action="store_true",
                        help="skip camera 6; camera 3 alone is ~3x cheaper")
    args = parser.parse_args()

    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import HfFileSystem

    record = RECORDS.get(args.record, args.record)
    fs = HfFileSystem()

    summary = json.loads(fs.cat_file(f"{REPO}/{record}/run_summary.json"))
    observations = summary["observations"]
    local_steps = sorted({o["local_step"] for o in observations})
    spacing = {b - a for a, b in zip(local_steps, local_steps[1:])}
    capture_type = "periodic" if len(spacing) == 1 else "mixed"
    print(f"{record}: {len(observations)} observations, "
          f"capture_type={capture_type} (local-step spacing {sorted(spacing)})")

    captures = [f"e{o['epoch']}_s{o['local_step']:06d}" for o in observations]
    if args.limit:
        captures, observations = captures[: args.limit], observations[: args.limit]

    rows, failures = [], []
    started = time.perf_counter()

    def work(pair):
        capture, observation = pair
        # One filesystem per thread: HfFileSystem holds per-instance HTTP
        # state and is not documented as thread-safe for concurrent seeks.
        local_fs = HfFileSystem()
        row = harvest_capture(
            local_fs, capture, {"capture_type": capture_type},
            with_boundary=not args.no_boundary, record=record,
        )
        return capture, observation, row

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, result in enumerate(
            pool.map(work, list(zip(captures, observations))), start=1
        ):
            capture, observation, row = result
            for field, expected in (("global_step", observation["global_step"]),
                                    ("dataset_row", observation["row_index"])):
                if row[field] != expected:
                    failures.append({"capture": capture,
                                     "error": f"{field} {row[field]} != {expected}"})
            rows.append(row)
            print(
                f"  [{index:>3}/{len(captures)}] {capture}  {row['capture_id']}  "
                f"grad {row['grad_norm_preclip']:6.3f} "
                f"clipmax {row['max_grad_norm']:.1f} "
                f"coef {row['clip_coefficient']:.3f}  "
                f"deliv {row['delivery']:.4f} cos {row['cosine']:.4f} "
                f"over {row['overshoot']:.4f} "
                f"zero_bf16 {row['zero_fraction_bf16']:.4f}",
                flush=True,
            )

    rows.sort(key=lambda r: (r["epoch"], r["local_step"]))

    with args.out.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    elapsed = time.perf_counter() - started
    runs = sorted({r["run_id"] for r in rows})
    over = [r["overshoot"] for r in rows if r["overshoot"] is not None]
    print(f"\nwrote {args.out}  ({len(rows)} rows, {len(failures)} failures) "
          f"in {elapsed / 60:.1f}m")
    print(f"run_id(s) present: {runs}")
    if len(runs) > 1:
        print("WARNING: more than one run_id in this file. Do not pool on step "
              "number without splitting by run_id first.")
    if over:
        ordered = sorted(over)
        print(f"overshoot: min {ordered[0]:.4f} median "
              f"{ordered[len(ordered) // 2]:.4f} max {ordered[-1]:.4f} "
              f"(1.0 == pure support truncation)")
    for failure in failures:
        print(f"  FAILURE {failure}")


if __name__ == "__main__":
    main()
