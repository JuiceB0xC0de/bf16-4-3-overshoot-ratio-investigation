"""Arm B driver: does the pooled overshoot follow the optimizer?

Modes
-----
``verify``  data pipeline only: reproduce the recorded token ids and counts.
``probe``   N steps, reporting the pooled intent norm per step so the
            scale-matching multiplier for SGD and Muon can be fixed exactly.
``run``     the real thing: N steps with a capture at every step.

The AdamW ``run`` at lr 2e-4, clip 1.0, from the pinned checkpoint over GSM8K
rows in order is the reference check.  It must reproduce the recorded step-100
behaviour or nothing else in this arm is interpretable.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import conditions as C
from data import StepData
from observer import DeliveryObserver, grad_stats, scalar_row

# Recorded values for run 20260810T091311Z (187M, clip 1.0) at optimizer step
# 100 / dataset row 99, from capture_summary_clip1.jsonl.
REFERENCE = {
    "step": 100,
    "loss": 1.4562201499938965,
    "grad_norm_preclip": 3.3906365638303937,
    "clip_coefficient": 0.29489556314235754,
    "whole_overshoot": 1.3257174657055293,
    "bf16_overshoot": 1.3257232653032234,
    "bf16_zero_fraction": 0.9385608528217241,
    "bf16_delivery": 0.4481422157385734,
    "whole_ideal_norm": 0.6923358714328083,
}
REFERENCE_TOL = {"overshoot": 0.02, "zero_fraction": 0.02}


def artifacts_dir() -> Path:
    d = Path(os.environ.get("SILICO_EXPERIMENT_ARTIFACTS_DIR", "./artifacts"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_loss(model, ids, logits_dtype: str):
    """Both loss variants from one forward; backward goes through the configured one.

    The recorded run's trainer was not supplied, so whether it upcast logits to
    fp32 before the cross entropy is a reconstruction choice.  Reporting both
    at every step lets the recorded step-100 loss settle it as a fact instead
    of a guess.
    """
    out = model(ids)
    logits = out.logits[0, :-1]
    targets = ids[0, 1:]
    loss_f32 = F.cross_entropy(logits.float(), targets)
    with torch.no_grad():
        loss_native = F.cross_entropy(logits, targets).float()
    return (loss_f32 if logits_dtype == "float32" else
            F.cross_entropy(logits, targets)), loss_f32, loss_native


def run(args) -> dict:
    device = "cuda"
    started = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    C.set_seed(args.seed)

    data = StepData()
    data_report = data.verify()
    data_report.update(data.window_stats(args.steps))
    print(f"[data] {json.dumps(data_report)}", flush=True)
    if args.mode == "verify":
        return {"mode": "verify", "data": data_report}

    model, raw_config = C.load_model(device=device)
    records = C.build_roster(model)
    summary = C.roster_summary(records)
    print(f"[roster] {json.dumps(summary)}", flush=True)

    condition = C.Condition(args.condition, records, lr=args.lr)
    observer = DeliveryObserver(records, condition, device)
    print(f"[condition] {json.dumps(condition.config())}", flush=True)

    wb = None
    if args.wandb and os.environ.get("WANDB_API_KEY"):
        try:
            import wandb

            wb = wandb.init(
                project=args.wandb_project,
                name=f"{args.condition}-{args.label}-lr{args.lr:g}",
                config={**condition.config(), "steps": args.steps,
                        "seed": args.seed, "label": args.label,
                        "model": C.MODEL_ID, "revision": C.MODEL_REVISION},
            )
            print(f"[wandb] {wb.url}", flush=True)
        except Exception as exc:                      # noqa: BLE001
            print(f"[wandb] disabled: {type(exc).__name__}: {exc}", flush=True)

    rows, captures, reference_check = [], {}, None
    full_steps = set(args.full_capture_steps or [])
    t0 = time.time()

    for step in range(1, args.steps + 1):
        row_index = data.row_index(step)
        ids = torch.tensor([data.ids_for_step(step)], device=device)

        condition.zero_grad()
        loss, loss_f32, loss_native = build_loss(model, ids, args.logits_dtype)
        loss.backward()

        pre_clip = grad_stats(records, C.CLIP_MAX_NORM)
        returned_norm = float(torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=C.CLIP_MAX_NORM))
        clip = {
            "loss_f32": float(loss_f32),
            "loss_native": float(loss_native),
            "grad_norm_preclip": returned_norm,
            "clip_fired": int(returned_norm > C.CLIP_MAX_NORM),
            **pre_clip,
        }

        observer.snapshot()
        condition.step()
        want_full = step in full_steps
        capture = observer.capture(want_geometry=True, want_per_tensor=want_full)

        row = scalar_row(step, row_index, float(loss_f32), capture, clip)
        rows.append(row)
        if want_full:
            captures[str(step)] = {"clip": clip, **capture}
        if wb is not None:
            wb.log({k: v for k, v in row.items() if isinstance(v, (int, float))},
                   step=step)

        if step == REFERENCE["step"] and args.condition == "adamw" and args.native:
            reference_check = check_reference(row)
            print(f"[reference] {json.dumps(reference_check)}", flush=True)

        if step % args.log_every == 0 or step == 1:
            rate = step / (time.time() - t0)
            print(
                f"[step {step}/{args.steps}] row={row_index} "
                f"loss={row['loss']:.4f} gnorm={returned_norm:.4f} "
                f"overshoot(whole)={row['whole/overshoot']:.6f} "
                f"overshoot(muon_set)={row['muon_covered/overshoot']:.6f} "
                f"zero_frac(bf16)={row['bf16/zero_fraction']:.6f} "
                f"| {rate:.2f} step/s",
                flush=True,
            )

    result = {
        "schema": "arm-b-optimizer-overshoot/1",
        "mode": args.mode,
        "condition": args.condition,
        "label": args.label,
        "lr": args.lr,
        "native": bool(args.native),
        "steps": args.steps,
        "seed": args.seed,
        "logits_dtype": args.logits_dtype,
        "model": {"id": C.MODEL_ID, "revision": C.MODEL_REVISION,
                  "config": raw_config},
        "data": data_report,
        "roster": summary,
        "condition_config": condition.config(),
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "elapsed_seconds": time.time() - started,
        "rows": rows,
        "full_captures": captures,
        "reference_check": reference_check,
        "wandb_url": wb.url if wb is not None else None,
    }
    if wb is not None:
        wb.finish()
    return result


def check_reference(row: dict) -> dict:
    """The gate: a reimplemented instrument that misses this measures nothing."""
    checks = {
        "loss": (row["loss"], REFERENCE["loss"], 0.05),
        "grad_norm_preclip": (row["grad_norm_preclip"],
                              REFERENCE["grad_norm_preclip"], 0.15),
        "whole_overshoot": (row["whole/overshoot"],
                            REFERENCE["whole_overshoot"], REFERENCE_TOL["overshoot"]),
        "bf16_zero_fraction": (row["bf16/zero_fraction"],
                               REFERENCE["bf16_zero_fraction"],
                               REFERENCE_TOL["zero_fraction"]),
        "bf16_delivery": (row["bf16/delivery"], REFERENCE["bf16_delivery"], 0.05),
    }
    out = {"passed": True, "checks": {}}
    for name, (got, want, tol) in checks.items():
        ok = abs(got - want) <= tol
        out["checks"][name] = {"observed": got, "recorded": want,
                               "abs_gap": abs(got - want), "tol": tol, "ok": ok}
        # Loss and gradient norm are trajectory-sensitive diagnostics; the plan's
        # stated gate is overshoot and zero fraction.
        if not ok and name in ("whole_overshoot", "bf16_zero_fraction"):
            out["passed"] = False
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["verify", "probe", "run"], default="run")
    ap.add_argument("--condition", choices=["adamw", "sgd", "muon"], default="adamw")
    ap.add_argument("--label", default="native",
                    help="native | scale_matched, recorded in the output")
    ap.add_argument("--native", action="store_true",
                    help="this is the ecologically real learning rate")
    ap.add_argument("--lr", type=float, default=C.ADAMW_LR)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--logits-dtype", choices=["float32", "native"], default="float32")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--full-capture-steps", type=int, nargs="*",
                    default=[1, 10, 50, 100, 200, 300])
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="mamba3-arm-b-optimizer-overshoot")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    result = run(args)
    name = args.out or f"armb_{args.condition}_{args.label}.json"
    path = artifacts_dir() / name
    path.write_text(json.dumps(result, indent=1))
    print(f"[done] wrote {path} ({path.stat().st_size/1e6:.2f} MB)", flush=True)

    jsonl = path.with_suffix(".rows.jsonl")
    with jsonl.open("w") as fh:
        for row in result.get("rows", []):
            fh.write(json.dumps(row) + "\n")
    print(f"[done] wrote {jsonl}", flush=True)

    ref = result.get("reference_check")
    if ref is not None and not ref["passed"]:
        raise SystemExit("REFERENCE CHECK FAILED: " + json.dumps(ref))


if __name__ == "__main__":
    main()
