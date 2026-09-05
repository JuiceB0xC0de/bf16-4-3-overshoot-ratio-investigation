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
# A run whose loss does not fall over 300 steps is not an optimizer walking a
# training trajectory; it is an optimizer walking a diverging one.  The pooled
# overshoot is still well defined there, but it is measured against a different
# model state, so the row has to be read as a broken-training condition rather
# than as a matched comparison.  Fixed before any 300-step run was analysed.
TRAJECTORY_LOSS_DROP_MIN = 0.10
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


def load_runs(inputs: Path, min_steps: int = 100) -> dict:
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
        if data.get("mode") != "run" or data.get("steps", 0) < min_steps:
            continue
        runs[f"{data['condition']}_{data['label']}"] = data
    return runs


def trajectory_health(rows: list) -> dict:
    """Did this condition actually train, or did it diverge at a matched norm?"""
    first = st.mean(r["loss"] for r in rows[:20])
    last = st.mean(r["loss"] for r in rows[-20:])
    drop = first - last
    finite = all(math.isfinite(r["loss"]) for r in rows)
    return {
        "loss_first20_mean": first,
        "loss_last20_mean": last,
        "loss_drop": drop,
        "loss_min": min(r["loss"] for r in rows),
        "loss_max": max(r["loss"] for r in rows),
        "all_losses_finite": finite,
        "grad_norm_preclip_median": st.median(r["grad_norm_preclip"] for r in rows),
        "threshold_loss_drop": TRAJECTORY_LOSS_DROP_MIN,
        "verdict": ("trains" if (finite and drop > TRAJECTORY_LOSS_DROP_MIN)
                    else "does_not_train"),
    }


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
            "clip_fired_every_step": all(r["clip_fired"] for r in run["rows"]),
            "clip_fired_first_100": sum(r["clip_fired"] for r in run["rows"][:100]),
            "trajectory": trajectory_health(run["rows"]),
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


def one_shadow_curve(run: dict) -> dict:
    """Overshoot vs intended-update magnitude, three optimizers, one trajectory."""
    rows = run["shadow_rows"][WARMUP:]
    optimizers = sorted(rows[0]["scales"])
    multipliers = sorted(rows[0]["scales"][optimizers[0]], key=lambda s: float(s))
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
        "source_run": f"{run['condition']}_{run['label']}",
        "host_optimizer": run["condition"],
        "host_lr": run["lr"],
        "n_steps": len(rows),
        "warmup_dropped": WARMUP,
        "multipliers": [float(m) for m in multipliers],
        "curve": curve,
        "at_adamw_magnitude": at_reference,
        "paired_difference_at_adamw_magnitude": paired,
        "self_check": run.get("shadow_self_check"),
    }


def matched_delivery(src: dict, grid=(0.03, 0.05, 0.09, 0.15, 0.25)) -> dict:
    """Compare the rules at a matched *delivered fraction*, not a matched norm.

    Matching the update norm still lets one rule move fewer coordinates than
    another, so a difference in overshoot could be read as "SGD simply takes
    smaller effective steps".  Interpolating each rule's curve to a common
    fraction of coordinates that actually moved removes that reading: what is
    left is the shape of the update distribution near the rounding threshold.
    """
    out = {"delivered_fraction_grid": list(grid), "at": {}}
    for target in grid:
        point = {}
        for name, pts in src["curve"].items():
            xs = [1.0 - p["zero_fraction"]["point"] for p in pts]
            ys = [p["overshoot"]["point"] for p in pts]
            pairs = sorted(zip(xs, ys))
            xs = [a for a, _ in pairs]
            ys = [b for _, b in pairs]
            if not (xs[0] <= target <= xs[-1]):
                continue                    # do not extrapolate past the sweep
            i = max(i for i in range(len(xs)) if xs[i] <= target)
            if i == len(xs) - 1:
                point[name] = ys[-1]
                continue
            lx, hx = math.log10(xs[i]), math.log10(xs[i + 1])
            w = (math.log10(target) - lx) / (hx - lx) if hx > lx else 0.0
            point[name] = ys[i] + w * (ys[i + 1] - ys[i])
        if "adamw" in point:
            for name in list(point):
                if name != "adamw":
                    point[f"{name}_minus_adamw"] = point[name] - point["adamw"]
        out["at"][f"{target:g}"] = point
    return out


def shadow_curve(runs: dict) -> dict:
    """One shadow curve per host trajectory.

    Every condition carries the shadow, so the same three update rules are
    compared at matched magnitude on six different model states.  A difference
    that survives all six is not a property of where AdamW happened to be.
    """
    sources = {k: one_shadow_curve(r) for k, r in sorted(runs.items())
               if r.get("shadow_rows")}
    if not sources:
        return {}
    primary = ("adamw_native" if "adamw_native" in sources
               else sorted(sources)[0])
    across = {}
    for opt in ("sgd", "muon"):
        key = f"{opt}_minus_adamw"
        per = {s: c["paired_difference_at_adamw_magnitude"][key]
               for s, c in sources.items()
               if key in c["paired_difference_at_adamw_magnitude"]}
        if per:
            across[key] = {
                "per_trajectory": per,
                "n_trajectories": len(per),
                "n_trajectories_negative": sum(v["point"] < 0 for v in per.values()),
                "n_trajectories_interval_excludes_zero": sum(
                    v["hi"] < 0 or v["lo"] > 0 for v in per.values()),
                "min_point": min(v["point"] for v in per.values()),
                "max_point": max(v["point"] for v in per.values()),
            }
    for c in sources.values():
        c["matched_delivery"] = matched_delivery(c)
    return {"primary": primary, "sources": sources,
            "across_trajectories": across, **sources[primary]}


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


def mechanism_points(runs: dict) -> dict:
    """Split each measured overshoot into its one-step and multi-step regimes.

    Writing N for the delivered squared norm and D for the inner product, the
    pooled overshoot N/D splits exactly over coordinates that moved exactly one
    lattice step and those that moved two or more:

        1/rho = (1 - s)/rho_1 + s/rho_2,   s = N_multi / N

    ``s`` and ``rho_1`` are measured directly (``multi_step_share_of_numerator``
    and ``single_step_only_ratio``); ``rho_2`` follows from the identity.  If
    the optimizers differ only through ``s``, every point should fall on one
    curve when plotted against it, whatever rule produced the update.
    """
    pts = []

    def add(source, optimizer, multiplier, step, geom):
        s = geom.get("multi_step_share_of_numerator")
        r1 = geom.get("single_step_only_ratio")
        rho = geom.get("ratio_from_realized_steps")
        if s is None or r1 is None or rho is None:
            return
        if not all(map(math.isfinite, (s, r1, rho))) or rho <= 0 or r1 <= 0:
            return
        rest = 1.0 / rho - (1.0 - s) / r1
        r2 = s / rest if s > 0 and rest > 0 else float("nan")
        pts.append({
            "source": source, "optimizer": optimizer, "multiplier": multiplier,
            "step": step, "kind": "run" if multiplier is None else "shadow",
            "multi_step_energy_share": s, "one_step_ratio": r1,
            "multi_step_ratio": r2, "overshoot": rho,
            "zero_step_fraction": geom.get("zero_step_fraction"),
            "mean_abs_step_of_moved": geom.get("mean_abs_step_of_moved"),
        })

    for key, run in sorted(runs.items()):
        for step, cap in sorted(run.get("full_captures", {}).items(),
                                key=lambda kv: int(kv[0])):
            # Same warm-up exclusion as every other table: AdamW's first steps
            # run before its second-moment estimate has settled.
            if int(step) <= WARMUP:
                continue
            geom = cap.get("geometry", {}).get("muon_covered")
            if geom:
                add(key, run["condition"], None, int(step), geom)
        for row in run.get("shadow_rows", []):
            if row["step"] <= WARMUP:
                continue
            for opt, scales in row.get("scales", {}).items():
                for mult, entry in scales.items():
                    if "geometry" in entry:
                        add(key, opt, float(mult), row["step"], entry["geometry"])
    return {
        "identity": "1/rho = (1-s)/rho_1 + s/rho_2, s = multi-step share of "
                    "delivered squared norm",
        "n_points": len(pts),
        "n_shadow_points": sum(p["kind"] == "shadow" for p in pts),
        "one_step_ratio_range": [min(p["one_step_ratio"] for p in pts),
                                 max(p["one_step_ratio"] for p in pts)] if pts else [],
        "multi_step_ratio_range": [
            min(p["multi_step_ratio"] for p in pts if p["multi_step_ratio"] == p["multi_step_ratio"]),
            max(p["multi_step_ratio"] for p in pts if p["multi_step_ratio"] == p["multi_step_ratio"]),
        ] if pts else [],
        "points": pts,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default=None,
                    help="directory of armb_*.json (default: artifacts dir)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--min-steps", type=int, default=100,
                    help="ignore short probe/sweep records")
    args = ap.parse_args()

    import os
    inputs = Path(args.inputs or os.environ.get(
        "SILICO_EXPERIMENT_ARTIFACTS_DIR", "."))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    runs = load_runs(inputs, args.min_steps)
    print(f"loaded {len(runs)} runs: {sorted(runs)}")
    if not runs:
        raise SystemExit(f"no arm-b run records found under {inputs}")

    table = condition_table(runs)
    curve = shadow_curve(runs)
    hists = step_histograms(runs)
    mech = mechanism_points(runs)
    (out / "mechanism.json").write_text(json.dumps(mech, indent=1))
    print(f"\nmechanism decomposition: {mech['n_points']} points "
          f"({mech['n_shadow_points']} from the shadow sweep)")
    if mech["points"]:
        print(f"  one-step regime ratio  {mech['one_step_ratio_range'][0]:.4f}"
              f" to {mech['one_step_ratio_range'][1]:.4f}")
        print(f"  multi-step regime ratio {mech['multi_step_ratio_range'][0]:.4f}"
              f" to {mech['multi_step_ratio_range'][1]:.4f}")

    (out / "by_condition.json").write_text(json.dumps(
        {"recorded_reference": RECORDED, "warmup_dropped": WARMUP,
         "rows": table}, indent=1))
    (out / "shadow_curve.json").write_text(json.dumps(curve, indent=1))
    (out / "step_histograms.json").write_text(json.dumps(hists, indent=1))

    # Flat CSV of the headline comparison, for anyone who wants a spreadsheet.
    lines = ["run,condition,label,lr,n_steps,"
             "muon_overshoot,muon_lo,muon_hi,whole_overshoot,whole_lo,whole_hi,"
             "muon_intent_norm,muon_zero_fraction,loss_last5,grad_norm_median,"
             "clip_fired_fraction,trajectory"]
    for r in table:
        m, w = r["muon_covered_overshoot"], r["whole_overshoot"]
        lines.append(
            f"{r['run']},{r['condition']},{r['label']},{r['lr']:g},{r['n_steps']},"
            f"{m['point']:.6f},{m['lo']:.6f},{m['hi']:.6f},"
            f"{w['point']:.6f},{w['lo']:.6f},{w['hi']:.6f},"
            f"{r['muon_covered_intent_norm']['point']:.6f},"
            f"{r['muon_covered_zero_fraction']['point']:.6f},"
            f"{r['loss_last5_mean']:.4f},{r['grad_norm_preclip_median']:.4f},"
            f"{r['clip_fired_fraction']:.4f},{r['trajectory']['verdict']}")
    (out / "by_condition.csv").write_text("\n".join(lines) + "\n")

    print(f"\n{'run':26s} {'lr':>9s} {'overshoot (Muon set)':>26s} {'intent':>9s} "
          f"{'loss':>7s}  {'clip':>5s} trajectory")
    for r in table:
        m = r["muon_covered_overshoot"]
        print(f"{r['run']:26s} {r['lr']:9.5g} "
              f"{m['point']:9.5f} [{m['lo']:.5f}, {m['hi']:.5f}] "
              f"{r['muon_covered_intent_norm']['point']:9.4f} "
              f"{r['loss_last5_mean']:7.3f}  "
              f"{r['clip_fired_fraction']:5.2f} {r['trajectory']['verdict']}")
    if curve:
        print(f"\nshadow control on {curve['primary']}, at AdamW's own magnitude:")
        for name, ci in curve["at_adamw_magnitude"].items():
            print(f"  {name:8s} {ci['point']:.5f} [{ci['lo']:.5f}, {ci['hi']:.5f}]")
        for name, ci in curve["paired_difference_at_adamw_magnitude"].items():
            print(f"  {name:24s} {ci['point']:+.5f} [{ci['lo']:+.5f}, {ci['hi']:+.5f}]"
                  f"  ({ci['n_steps_negative']}/{ci['n_steps']} steps negative)")
        print(f"  self-check: {curve['self_check']}")
        md = curve["sources"][curve["primary"]]["matched_delivery"]
        print("\nat a matched fraction of coordinates delivered "
              "(interpolated along each rule's own curve):")
        print(f"  {'delivered':>9s} {'adamw':>8s} {'muon':>8s} {'sgd':>8s} "
              f"{'muon-adamw':>11s} {'sgd-adamw':>11s}")
        for frac, pt in md["at"].items():
            if "adamw" not in pt:
                continue
            print(f"  {frac:>9s} {pt['adamw']:8.5f} "
                  f"{pt.get('muon', float('nan')):8.5f} "
                  f"{pt.get('sgd', float('nan')):8.5f} "
                  f"{pt.get('muon_minus_adamw', float('nan')):+11.5f} "
                  f"{pt.get('sgd_minus_adamw', float('nan')):+11.5f}")
        print("\nsame comparison replicated on every host trajectory:")
        for key, agg in curve.get("across_trajectories", {}).items():
            print(f"  {key:24s} range [{agg['min_point']:+.5f}, {agg['max_point']:+.5f}]"
                  f"  {agg['n_trajectories_negative']}/{agg['n_trajectories']} negative,"
                  f" {agg['n_trajectories_interval_excludes_zero']} intervals exclude 0")
            for src, ci in agg["per_trajectory"].items():
                print(f"      {src:24s} {ci['point']:+.5f} "
                      f"[{ci['lo']:+.5f}, {ci['hi']:+.5f}]")
    print(f"\nwrote {out}/by_condition.json, shadow_curve.json, "
          f"step_histograms.json, by_condition.csv")


if __name__ == "__main__":
    main()
