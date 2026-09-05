"""Join the per-condition records into plot-ready tables.

Two questions, two tables.

``by_condition.json`` -- the planned comparison.  One row per run: the pooled
overshoot on each coordinate subset, the achieved intent magnitude, the
trajectory health, and the lattice geometry.  Each optimizer walks its own
trajectory here, so a difference between rows confounds the update
distribution with the model state the run reached.

``shadow_curve.json`` -- the control.  Overshoot as a function of intended
update magnitude, for all three optimizers, on one common trajectory with
identical gradients and an identical lattice.  This is the table that can
separate "the optimizer's update distribution" from "how big its steps were".

Bootstrap intervals are over steps, resampling whole steps, because
consecutive steps of one run are not independent draws in any other sense.

    python3 src/analyze.py --inputs <dir> --out results
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path

import numpy as np

SUBSETS = ("whole", "bf16_recorded", "bf16_true", "muon_covered")
WARMUP = 5          # AdamW's first steps are bias-correction transients
RECORDED = {
    "step100_whole_overshoot": 1.3257174657055293,
    "step100_bf16_zero_fraction": 0.9385608528217241,
    "clip1_median_overshoot": 1.333721,
    "clip8_median_overshoot": 1.334682,
    "uniform_x_reference": 4.0 / 3.0,
}


def boot_ci(values, draws: int = 2000, seed: int = 42, lo: float = 2.5,
            hi: float = 97.5) -> dict:
    arr = np.asarray([v for v in values if v == v], dtype=np.float64)
    if arr.size == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(draws, arr.size))
    meds = np.median(arr[idx], axis=1)
    return {"point": float(np.median(arr)), "lo": float(np.percentile(meds, lo)),
            "hi": float(np.percentile(meds, hi)), "n": int(arr.size)}


def load_runs(inputs: Path) -> dict:
    runs = {}
    for path in sorted(inputs.glob("armb_*.json")):
        if ".rows." in path.name:
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if data.get("schema") != "arm-b-optimizer-overshoot/1":
            continue
        if data.get("mode") != "run" or data.get("steps", 0) < 100:
            continue
        runs[f"{data['condition']}_{data['label']}"] = data
    return runs


def condition_table(runs: dict) -> list:
    rows = []
    for key, run in sorted(runs.items()):
        steps = run["rows"][WARMUP:]
        entry = {
            "run": key,
            "condition": run["condition"],
            "label": run["label"],
            "lr": run["lr"],
            "n_steps": len(run["rows"]),
            "n_steps_used": len(steps),
            "warmup_dropped": WARMUP,
            "config": run["condition_config"],
            "loss_first": run["rows"][0]["loss"],
            "loss_last5_mean": st.mean(r["loss"] for r in run["rows"][-5:]),
            "grad_norm_preclip_median": st.median(r["grad_norm_preclip"] for r in steps),
            "grad_norm_preclip_max": max(r["grad_norm_preclip"] for r in steps),
            "clip_fired_fraction": st.mean(r["clip_fired"] for r in run["rows"]),
        }
        for subset in SUBSETS:
            entry[f"{subset}_overshoot"] = boot_ci([r[f"{subset}/overshoot"] for r in steps])
            entry[f"{subset}_zero_fraction"] = boot_ci(
                [r[f"{subset}/zero_fraction"] for r in steps])
            entry[f"{subset}_intent_norm"] = boot_ci([r[f"{subset}/ideal"] for r in steps])
            entry[f"{subset}_delivery"] = boot_ci([r[f"{subset}/delivery"] for r in steps])
        for subset in ("bf16_true", "muon_covered"):
            for field in ("x_median", "zero_step_fraction",
                          "abs_step_ge_2_share_of_moved", "mean_abs_step_of_moved",
                          "ratio_predicted_from_x", "decay_x_max"):
                entry[f"geom_{subset}_{field}"] = boot_ci(
                    [r[f"geom_{subset}/{field}"] for r in steps])
        if run.get("reference_check"):
            entry["reference_check"] = run["reference_check"]
        if run.get("shadow_self_check"):
            entry["shadow_self_check"] = run["shadow_self_check"]
        rows.append(entry)
    return rows


def shadow_curve(runs: dict) -> dict:
    """Overshoot vs intended-update magnitude, three optimizers, one trajectory."""
    source = next((r for r in runs.values() if r.get("shadow_rows")), None)
    if source is None:
        return {}
    rows = source["shadow_rows"][WARMUP:]
    optimizers = sorted(rows[0]["scales"])
    multipliers = sorted(rows[0]["scales"][optimizers[0]],
                         key=lambda s: float(s))
    curve = {}
    for name in optimizers:
        points = []
        for mult in multipliers:
            overs = [r["scales"][name][mult]["overshoot"] for r in rows]
            zeros = [r["scales"][name][mult]["zero_fraction"] for r in rows]
            targets = [r["scales"][name][mult]["target_intent_norm"] for r in rows]
            points.append({
                "multiplier": float(mult),
                "intent_norm_median": st.median(targets),
                "overshoot": boot_ci(overs),
                "zero_fraction": boot_ci(zeros),
            })
        curve[name] = points

    ref = "1"
    at_reference = {
        name: next(p for p in pts if p["multiplier"] == 1.0)["overshoot"]
        for name, pts in curve.items()
    }
    # Paired per-step differences: the shadow shares a trajectory, so each step
    # gives a matched triple and the difference is paired, not two samples.
    paired = {}
    for name in optimizers:
        if name == "adamw":
            continue
        diffs = [r["scales"][name][ref]["overshoot"] - r["scales"]["adamw"][ref]["overshoot"]
                 for r in rows]
        paired[f"{name}_minus_adamw"] = boot_ci(diffs)
        paired[f"{name}_minus_adamw"]["n_steps_negative"] = int(sum(d < 0 for d in diffs))
        paired[f"{name}_minus_adamw"]["n_steps"] = len(diffs)
    return {
        "source_run": f"{source['condition']}_{source['label']}",
        "n_steps": len(rows),
        "warmup_dropped": WARMUP,
        "multipliers": [float(m) for m in multipliers],
        "curve": curve,
        "at_adamw_magnitude": at_reference,
        "paired_difference_at_adamw_magnitude": paired,
        "self_check": source.get("shadow_self_check"),
    }


def step_histograms(runs: dict) -> dict:
    """Realized move size in ULPs, per condition, at the last full capture.

    Kept as the full distribution rather than a zero/nonzero split: Arm A found
    35.88% of nonzero moves are two or more ULPs, which is where an optimizer
    difference has to show up if it exists.
    """
    out = {}
    for key, run in sorted(runs.items()):
        caps = run.get("full_captures", {})
        if not caps:
            continue
        last = max(caps, key=lambda s: int(s))
        geo = caps[last].get("geometry", {}).get("muon_covered")
        if not geo:
            continue
        hist = geo["step_hist"]
        total = sum(hist["counts"]) + hist["above_edges"]
        out[key] = {
            "step": int(last),
            "edges": hist["edges"],
            "counts": hist["counts"],
            "above_edges": hist["above_edges"],
            "total": total,
            "share": [c / total for c in hist["counts"]],
            "zero_step_fraction": geo["zero_step_fraction"],
            "abs_step_ge_2_share_of_moved": geo["abs_step_ge_2_share_of_moved"],
            "mean_abs_step_of_moved": geo["mean_abs_step_of_moved"],
            "x_log10_hist": geo["x_log10_hist"],
            "x_quantiles_from_hist": geo["x_quantiles_from_hist"],
            "chi_square_vs_uniform": geo["x_in_unit_interval_hist"]["chi_square_vs_uniform"],
            "ratio_predicted_from_x": geo["ratio_predicted_from_x"],
            "ratio_from_realized_steps": geo["ratio_from_realized_steps"],
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default=None,
                    help="directory of armb_*.json (default: artifacts dir)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    import os
    inputs = Path(args.inputs or os.environ.get(
        "SILICO_EXPERIMENT_ARTIFACTS_DIR", "."))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    runs = load_runs(inputs)
    print(f"loaded {len(runs)} runs: {sorted(runs)}")
    if not runs:
        raise SystemExit(f"no arm-b run records found under {inputs}")

    table = condition_table(runs)
    curve = shadow_curve(runs)
    hists = step_histograms(runs)

    (out / "by_condition.json").write_text(json.dumps(
        {"recorded_reference": RECORDED, "warmup_dropped": WARMUP,
         "rows": table}, indent=1))
    (out / "shadow_curve.json").write_text(json.dumps(curve, indent=1))
    (out / "step_histograms.json").write_text(json.dumps(hists, indent=1))

    # Flat CSV of the headline comparison, for anyone who wants a spreadsheet.
    lines = ["run,condition,label,lr,n_steps,"
             "muon_overshoot,muon_lo,muon_hi,whole_overshoot,whole_lo,whole_hi,"
             "muon_intent_norm,muon_zero_fraction,loss_last5,grad_norm_median"]
    for r in table:
        m, w = r["muon_covered_overshoot"], r["whole_overshoot"]
        lines.append(
            f"{r['run']},{r['condition']},{r['label']},{r['lr']:g},{r['n_steps']},"
            f"{m['point']:.6f},{m['lo']:.6f},{m['hi']:.6f},"
            f"{w['point']:.6f},{w['lo']:.6f},{w['hi']:.6f},"
            f"{r['muon_covered_intent_norm']['point']:.6f},"
            f"{r['muon_covered_zero_fraction']['point']:.6f},"
            f"{r['loss_last5_mean']:.4f},{r['grad_norm_preclip_median']:.4f}")
    (out / "by_condition.csv").write_text("\n".join(lines) + "\n")

    print(f"\n{'run':26s} {'lr':>9s} {'overshoot (Muon set)':>26s} {'intent':>9s} {'loss':>7s}")
    for r in table:
        m = r["muon_covered_overshoot"]
        print(f"{r['run']:26s} {r['lr']:9.5g} "
              f"{m['point']:9.5f} [{m['lo']:.5f}, {m['hi']:.5f}] "
              f"{r['muon_covered_intent_norm']['point']:9.4f} "
              f"{r['loss_last5_mean']:7.3f}")
    if curve:
        print(f"\nshadow control, one trajectory, at AdamW's own magnitude:")
        for name, ci in curve["at_adamw_magnitude"].items():
            print(f"  {name:8s} {ci['point']:.5f} [{ci['lo']:.5f}, {ci['hi']:.5f}]")
        for name, ci in curve["paired_difference_at_adamw_magnitude"].items():
            print(f"  {name:24s} {ci['point']:+.5f} [{ci['lo']:+.5f}, {ci['hi']:+.5f}]"
                  f"  ({ci['n_steps_negative']}/{ci['n_steps']} steps negative)")
        print(f"  self-check: {curve['self_check']}")
    print(f"\nwrote {out}/by_condition.json, shadow_curve.json, "
          f"step_histograms.json, by_condition.csv")


if __name__ == "__main__":
    main()
