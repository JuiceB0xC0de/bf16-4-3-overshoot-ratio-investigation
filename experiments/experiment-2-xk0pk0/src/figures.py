"""Build the report figures from the analysis tables.

Reads ``results/by_condition.json``, ``results/shadow_curve.json`` and
``results/step_histograms.json`` and writes one figure bundle per display
under ``figures/``.  Nothing here recomputes a measurement; every value comes
from the analysis tables, so a figure and the prose beside it cannot drift.

    uv run --no-sync python src/figures.py
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import plotly.graph_objects as go
from silico_figures import (EDITORIAL_8, add_reference_line, apply_theme,
                            save_figure_bundle)

FOUR_THIRDS = 4.0 / 3.0
# AdamW is the optimizer the recorded run used, so it is the object of study and
# holds the emphasis colour in every figure it appears in.  Written once and
# extended, never reshuffled.
COLORS = {"adamw": EDITORIAL_8[0], "muon": EDITORIAL_8[1], "sgd": EDITORIAL_8[2]}
PRETTY = {"adamw": "AdamW", "sgd": "SGD", "muon": "Muon"}
LABEL = {"native": "native", "scale_matched": "scale-matched, probe",
         "scale_matched_v2": "scale-matched", "lr0.005": "lr scan",
         "lr0.01": "lr scan"}
# The two figures that draw one line per run stay readable only while at most
# two lines share a hue, so they show a fixed slice: the three runs matched to
# AdamW's update magnitude, plus each rule's own native setting.
PER_RUN_SUBSET = ("adamw_native", "muon_scale_matched_v2", "sgd_scale_matched_v2",
                  "muon_lr0.005", "sgd_native")
def _sup(k: int) -> str:
    digits = "⁰¹²³⁴⁵⁶⁷⁸⁹"
    body = "".join(digits[int(d)] for d in str(abs(k)))
    return ("⁻" if k < 0 else "") + body


# Decade ticks: these axes are dimensionless ratios, so Plotly's SI prefixes
# ("10n", "1u") would be nonsense.
DECADE_TICKS = {"tickvals": [10.0 ** k for k in range(-9, 5)],
                "ticktext": [("1" if k == 0 else f"10{_sup(k)}")
                             for k in range(-9, 5)]}


def _lighten(hex_color: str, amount: float) -> str:
    """Same hue, higher lightness: the scale-matched twin of an optimizer."""
    h = hex_color.lstrip("#")
    rgb = [int(h[i:i + 2], 16) for i in (0, 2, 4)]
    rgb = [int(round(c + (255 - c) * amount)) for c in rgb]
    return "#" + "".join(f"{c:02X}" for c in rgb)


def fig_mechanism(mech: dict) -> tuple:
    """Overshoot against the share of delivered energy in multi-step moves.

    Every measured point, from all runs and all 9 shadow magnitudes.  If the
    optimizers differed in some way beyond this one number, points of different
    colour at the same x would sit at different heights.
    """
    pts = [p for p in mech["points"] if p["multi_step_ratio"] == p["multi_step_ratio"]]
    if not pts:
        return None, []
    fig = go.Figure()
    data = []
    # One trace per optimizer, with the whole-run points carried as a larger
    # diamond inside the same trace: splitting them into their own traces would
    # put two identically-coloured series in the legend.
    for opt in ("sgd", "muon", "adamw"):
        sel = sorted((p for p in pts if p["optimizer"] == opt),
                     key=lambda p: p["kind"])          # 'run' before 'shadow'
        if not sel:
            continue
        is_run = [p["kind"] == "run" for p in sel]
        fig.add_trace(go.Scatter(
            x=[p["multi_step_energy_share"] for p in sel],
            y=[p["overshoot"] for p in sel], mode="markers", name=PRETTY[opt],
            marker=dict(color=COLORS[opt],
                        size=[11 if r else 6 for r in is_run],
                        symbol=["diamond" if r else "circle" for r in is_run],
                        opacity=[1.0 if r else 0.5 for r in is_run],
                        line=dict(width=[1.2 if r else 0 for r in is_run],
                                  color="#FFFFFF")),
            customdata=["whole run" if r else "shadow sweep" for r in is_run],
            hovertemplate=(f"{PRETTY[opt]}, %{{customdata}}"
                           "<br>multi-step energy share %{x:.4f}"
                           "<br>overshoot %{y:.4f}<extra></extra>")))
        data.extend({"optimizer": PRETTY[opt], "kind": p["kind"],
                     "multi_step_energy_share": p["multi_step_energy_share"],
                     "overshoot": p["overshoot"],
                     "one_step_ratio": p["one_step_ratio"],
                     "source": p["source"]} for p in sel)
    add_reference_line(fig, y=FOUR_THIRDS, label="4/3 = 1.333")
    fig.update_xaxes(title_text="Share of delivered squared norm in moves of 2+ ULPs")
    fig.update_yaxes(title_text="Pooled overshoot ‖r‖²/⟨i,r⟩")
    apply_theme(fig, height=450)
    return fig, data


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def series_color(condition: str, label: str) -> str:
    """A rule at its own learning rate keeps the optimizer's colour; the run
    matched to AdamW's update magnitude is its lighter twin, so the two never
    rely on position alone to be told apart."""
    base = COLORS[condition]
    return _lighten(base, 0.28) if label.startswith("scale_matched") else base


def series_name(condition: str, label: str) -> str:
    return f"{PRETTY[condition]}, {LABEL.get(label, label)}"


def load(results: Path) -> tuple:
    by = json.loads((results / "by_condition.json").read_text())
    curve = json.loads((results / "shadow_curve.json").read_text())
    hist = json.loads((results / "step_histograms.json").read_text())
    mech = json.loads((results / "mechanism.json").read_text())
    return by, curve, hist, mech


def write_entity_colors(root: Path, by: dict) -> None:
    """The colour every figure must use for each rule, written from the runs
    that actually exist so a stale name cannot outlive the run it named."""
    path = root / "figures" / "entity_colors.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {PRETTY[c]: COLORS[c] for c in ("adamw", "muon", "sgd")}
    for r in by["rows"]:
        out[series_name(r["condition"], r["label"])] = \
            series_color(r["condition"], r["label"])
    path.write_text(json.dumps(out, indent=1))


# --------------------------------------------------------------------------
# Figure 1 -- the planned six-run comparison.
# --------------------------------------------------------------------------
def fig_by_condition(by: dict, subset: str = "muon_covered") -> tuple:
    """Pooled overshoot per run, with the uniform-x idealisation as reference.

    Dot-and-interval rather than bars: overshoot has no meaningful zero (a
    delivered step of zero size makes the ratio undefined, not zero), and the
    bootstrap interval is the evidence.
    """
    fig = go.Figure()
    data = []
    # One trace per rule, not per run: nine legend entries would repeat the
    # y-axis, and several of them would have to share a hue.
    order: list = []
    for cond in ("sgd", "muon", "adamw"):        # AdamW drawn last, at the top
        rows = sorted((r for r in by["rows"] if r["condition"] == cond),
                      key=lambda r: -r[f"{subset}_intent_norm"]["point"])
        if not rows:
            continue
        names, mid, plus, minus, broke = [], [], [], [], []
        for r in rows:
            ci = r[f"{subset}_overshoot"]
            bad = r["trajectory"]["verdict"] != "trains"
            # Two lines: at 640px a single-line label with optimizer, setting
            # and learning rate eats half the panel width.
            name = (f"{series_name(cond, r['label'])}<br>lr {r['lr']:g}"
                    + (", did not train" if bad else ""))
            names.append(name)
            mid.append(ci["point"])
            plus.append(ci["hi"] - ci["point"])
            minus.append(ci["point"] - ci["lo"])
            broke.append(bad)
            data.append({"run": name, "condition": cond, "label": r["label"],
                         "lr": r["lr"], "overshoot": ci["point"],
                         "lo": ci["lo"], "hi": ci["hi"], "n_steps": ci["n"],
                         "intent_norm": r[f"{subset}_intent_norm"]["point"],
                         "trains": not bad})
        order.extend(names)
        fig.add_trace(go.Scatter(
            x=mid, y=names, mode="markers", name=PRETTY[cond], showlegend=False,
            marker=dict(color=COLORS[cond], size=13,
                        symbol=["circle-open" if b else "circle" for b in broke],
                        line=dict(width=2.5, color=COLORS[cond])),
            error_x=dict(type="data", symmetric=False, array=plus,
                         arrayminus=minus, color=COLORS[cond],
                         thickness=1.4, width=5),
            hovertemplate="%{y}<br>overshoot %{x:.4f}<extra></extra>"))
    # The y-axis already names the rule and the setting, so the only encoding
    # left to explain is the hollow marker.
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers", name="did not train",
        marker=dict(color="#31362E", size=13, symbol="circle-open",
                    line=dict(width=2.5, color="#31362E")),
        hoverinfo="skip"))
    add_reference_line(fig, x=FOUR_THIRDS, label="4/3 = 1.333")
    fig.update_xaxes(title_text="Pooled overshoot ‖r‖²/⟨i,r⟩")
    fig.update_yaxes(title_text="Optimizer, learning-rate setting",
                     type="category", categoryorder="array", categoryarray=order)
    apply_theme(fig, height=max(320, 150 + 46 * len(data)))
    return fig, data


# --------------------------------------------------------------------------
# Figure 1b -- the same nine runs read against how big their updates were.
# --------------------------------------------------------------------------
def fig_runs_vs_magnitude(by: dict, subset: str = "muon_covered") -> tuple:
    """Overshoot against intended update norm, one point per real training run.

    The shadow control shows the magnitude dependence on a single trajectory.
    This is the same relationship in the nine real runs, each with its own
    trajectory, and it separates the two readings: overshoot falls with update
    size for every rule, and SGD sits below the other two at every size.
    """
    fig = go.Figure()
    data = []
    for cond in ("sgd", "muon", "adamw"):
        rows = sorted((r for r in by["rows"] if r["condition"] == cond),
                      key=lambda r: r[f"{subset}_intent_norm"]["point"])
        if not rows:
            continue
        xs = [r[f"{subset}_intent_norm"]["point"] for r in rows]
        ys = [r[f"{subset}_overshoot"]["point"] for r in rows]
        broke = [r["trajectory"]["verdict"] != "trains" for r in rows]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers" if len(xs) > 1 else "markers",
            name=PRETTY[cond], line=dict(color=COLORS[cond], width=2.2),
            marker=dict(color=COLORS[cond], size=12,
                        symbol=["circle-open" if b else "circle" for b in broke],
                        line=dict(width=2.4, color=COLORS[cond])),
            error_y=dict(type="data", symmetric=False,
                         array=[r[f"{subset}_overshoot"]["hi"] - y
                                for r, y in zip(rows, ys)],
                         arrayminus=[y - r[f"{subset}_overshoot"]["lo"]
                                     for r, y in zip(rows, ys)],
                         color=COLORS[cond], thickness=1.2, width=3),
            customdata=[f"lr {r['lr']:g}" + (", did not train" if b else "")
                        for r, b in zip(rows, broke)],
            hovertemplate=(f"{PRETTY[cond]}, %{{customdata}}"
                           "<br>intent norm %{x:.4g}"
                           "<br>overshoot %{y:.4f}<extra></extra>")))
        data.extend({"optimizer": PRETTY[cond], "run": r["run"], "lr": r["lr"],
                     "intent_norm": x, "overshoot": y,
                     "lo": r[f"{subset}_overshoot"]["lo"],
                     "hi": r[f"{subset}_overshoot"]["hi"], "trains": not b}
                    for r, x, y, b in zip(rows, xs, ys, broke))
    # No vertical marker for AdamW's magnitude: its point and the two
    # scale-matched points already sit on the same x to four figures, and a
    # second reference label collides with the 4/3 one in the top corner.
    add_reference_line(fig, y=FOUR_THIRDS, label="4/3 = 1.333")
    fig.update_xaxes(title_text="Intended update norm (log)", type="log",
                     **DECADE_TICKS)
    fig.update_yaxes(title_text="Pooled overshoot ‖r‖²/⟨i,r⟩")
    apply_theme(fig, height=440)
    return fig, data


# --------------------------------------------------------------------------
# Figure 2 -- the confound-free control.
# --------------------------------------------------------------------------
def fig_shadow_curve(curve: dict) -> tuple:
    """Overshoot against intended update magnitude on one shared trajectory."""
    src = curve["sources"][curve["primary"]]
    fig = go.Figure()
    data = []
    for name in ("sgd", "muon", "adamw"):          # AdamW drawn last, on top
        pts = src["curve"][name]
        xs = [p["intent_norm_median"] for p in pts]
        lo = [p["overshoot"]["lo"] for p in pts]
        hi = [p["overshoot"]["hi"] for p in pts]
        mid = [p["overshoot"]["point"] for p in pts]
        # The bootstrap interval rides on the line itself as error bars rather
        # than as a filled band: at this width three translucent bands overlap
        # into mud, and these intervals are narrow enough to read as ticks.
        fig.add_trace(go.Scatter(
            x=xs, y=mid, mode="lines+markers", name=PRETTY[name],
            line=dict(color=COLORS[name], width=2.6),
            marker=dict(color=COLORS[name], size=7),
            error_y=dict(type="data", symmetric=False,
                         array=[h - m for h, m in zip(hi, mid)],
                         arrayminus=[m - l for m, l in zip(mid, lo)],
                         color=COLORS[name], thickness=1.2, width=3),
            hovertemplate=(f"{PRETTY[name]}<br>intent norm %{{x:.4g}}"
                           "<br>overshoot %{y:.4f}<extra></extra>")))
        for p in pts:
            data.append({"optimizer": PRETTY[name], "multiplier": p["multiplier"],
                         "intent_norm": p["intent_norm_median"],
                         "overshoot": p["overshoot"]["point"],
                         "lo": p["overshoot"]["lo"], "hi": p["overshoot"]["hi"]})
    add_reference_line(fig, y=FOUR_THIRDS, label="4/3 = 1.333")
    fig.update_xaxes(title_text="Intended update norm (log)", type="log",
                     **DECADE_TICKS)
    fig.update_yaxes(title_text="Pooled overshoot ‖r‖²/⟨i,r⟩")
    apply_theme(fig, height=440)
    return fig, data


# --------------------------------------------------------------------------
# Figure 2b -- the same sweep read against how much actually moved.
# --------------------------------------------------------------------------
def fig_shadow_vs_delivery(curve: dict) -> tuple:
    """Overshoot against the fraction of coordinates that moved at all.

    Matching the update norm leaves open the reading that SGD simply takes
    smaller effective steps.  On this axis that reading is closed: at the same
    fraction of coordinates delivered, the rules still separate.
    """
    src = curve["sources"][curve["primary"]]
    fig = go.Figure()
    data = []
    for name in ("sgd", "muon", "adamw"):
        pts = src["curve"][name]
        xs = [1.0 - p["zero_fraction"]["point"] for p in pts]
        ys = [p["overshoot"]["point"] for p in pts]
        lo = [p["overshoot"]["lo"] for p in pts]
        hi = [p["overshoot"]["hi"] for p in pts]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers", name=PRETTY[name],
            line=dict(color=COLORS[name], width=2.6),
            marker=dict(color=COLORS[name], size=7),
            error_y=dict(type="data", symmetric=False,
                         array=[h - m for h, m in zip(hi, ys)],
                         arrayminus=[m - l for m, l in zip(ys, lo)],
                         color=COLORS[name], thickness=1.2, width=3),
            hovertemplate=(f"{PRETTY[name]}<br>delivered %{{x:.4f}}"
                           "<br>overshoot %{y:.4f}<extra></extra>")))
        for x, y, l, h in zip(xs, ys, lo, hi):
            data.append({"optimizer": PRETTY[name], "delivered_fraction": x,
                         "overshoot": y, "lo": l, "hi": h})
    add_reference_line(fig, y=FOUR_THIRDS, label="4/3 = 1.333")
    fig.update_xaxes(
        title_text="Coordinates that moved (log)", type="log",
        tickvals=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5],
        ticktext=["0.1%", "0.3%", "1%", "3%", "10%", "30%", "50%"])
    fig.update_yaxes(title_text="Pooled overshoot ‖r‖²/⟨i,r⟩")
    apply_theme(fig, height=440)
    return fig, data


# --------------------------------------------------------------------------
# Figure 3 -- does the gap survive a change of trajectory?
# --------------------------------------------------------------------------
def fig_shadow_replication(curve: dict) -> tuple:
    """Paired per-step difference from AdamW, at matched magnitude, per host."""
    across = curve.get("across_trajectories", {})
    if not across:
        return None, []
    fig = go.Figure()
    data = []
    order = sorted(curve["sources"])
    for key, opt in (("sgd_minus_adamw", "sgd"), ("muon_minus_adamw", "muon")):
        agg = across.get(key)
        if not agg:
            continue
        xs, ys, plus, minus = [], [], [], []
        for src in order:
            ci = agg["per_trajectory"].get(src)
            if ci is None:
                continue
            host = curve["sources"][src]
            label = (f"{PRETTY[host['host_optimizer']]}, "
                     f"{LABEL.get(src.split('_', 1)[1], src)}")
            xs.append(ci["point"])
            ys.append(label)
            plus.append(ci["hi"] - ci["point"])
            minus.append(ci["point"] - ci["lo"])
            data.append({"comparison": f"{PRETTY[opt]} − AdamW",
                         "host_trajectory": label, "difference": ci["point"],
                         "lo": ci["lo"], "hi": ci["hi"],
                         "n_steps": ci["n"],
                         "n_steps_negative": ci["n_steps_negative"]})
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers", name=f"{PRETTY[opt]} − AdamW",
            marker=dict(color=COLORS[opt], size=12),
            error_x=dict(type="data", symmetric=False, array=plus,
                         arrayminus=minus, color=COLORS[opt],
                         thickness=1.4, width=5),
            hovertemplate=(f"{PRETTY[opt]} − AdamW<br>%{{y}}"
                           "<br>difference %{x:+.4f}<extra></extra>")))
    add_reference_line(fig, x=0.0, label="no difference")
    fig.update_xaxes(title_text="Overshoot difference from AdamW at matched magnitude")
    fig.update_yaxes(title_text="Host trajectory the shadow rode on")
    apply_theme(fig, height=420)
    return fig, data


# --------------------------------------------------------------------------
# Figure 4 -- the mechanism: how far each coordinate actually moved.
# --------------------------------------------------------------------------
def fig_step_histogram(hist: dict, by: dict, max_ulp: int = 6) -> tuple:
    """Share of coordinates that moved k lattice steps, per condition.

    Drawn as connected markers rather than bars: the share spans four decades,
    which needs a log axis, and a bar on a log axis no longer encodes its value
    by length.  The move count is an ordered integer axis, so a line is a fair
    reading of it.
    """
    meta = {r["run"]: r for r in by["rows"]}
    fig = go.Figure()
    data = []
    keys = [k for k in PER_RUN_SUBSET if k in hist]
    bins = list(range(max_ulp + 1))
    cats = [str(b) for b in bins] + [f"≥{max_ulp + 1}"]
    for key in keys:
        h = hist[key]
        share = h["share"]
        ys = [share[k] for k in bins]
        ys.append(sum(share[max_ulp + 1:]) + h["above_edges"] / h["total"])
        r = meta.get(key, {})
        cond, label = r.get("condition", key.split("_")[0]), r.get("label", "native")
        fig.add_trace(go.Scatter(
            x=cats, y=ys, mode="lines+markers",
            name=series_name(cond, label),
            line=dict(color=series_color(cond, label), width=2.4,
                      dash="dot" if label.startswith("scale_matched") else "solid"),
            marker=dict(color=series_color(cond, label), size=8,
                        symbol=("diamond" if label.startswith("scale_matched")
                                else "circle")),
            hovertemplate="%{fullData.name}<br>%{x} ULP: %{y:.5f}<extra></extra>"))
        for b, y in zip(cats, ys):
            data.append({"run": key, "optimizer": PRETTY[cond], "label": label,
                         "ulp_step": b, "share": y})
    fig.update_xaxes(title_text="Realized move (lattice steps, ULPs)", type="category")
    fig.update_yaxes(title_text="Share of coordinates (log)", type="log")
    apply_theme(fig, height=430)
    return fig, data


# --------------------------------------------------------------------------
# Figure 5 -- why 4/3 is only an idealisation.
# --------------------------------------------------------------------------
def fig_x_distribution(hist: dict, by: dict) -> tuple:
    """Distribution of x = |intended update| / ULP, per condition.

    The 4/3 closed form assumes x is uniform on the unit interval.  This is the
    measured x, pooled over the Muon-covered set at the last full capture.
    """
    meta = {r["run"]: r for r in by["rows"]}
    fig = go.Figure()
    data = []
    keys = [k for k in PER_RUN_SUBSET if k in hist]
    for key in keys:
        h = hist[key]["x_log10_hist"]
        edges, counts = h["edges"], h["counts"]
        total = sum(counts) + h["below"] + h["above"]
        if not total:
            continue
        centres = [10.0 ** (0.5 * (edges[i] + edges[i + 1]))
                   for i in range(len(counts))]
        share = [c / total for c in counts]
        r = meta.get(key, {})
        cond, label = r.get("condition", key.split("_")[0]), r.get("label", "native")
        fig.add_trace(go.Scatter(
            x=centres, y=share, mode="lines",
            name=series_name(cond, label),
            line=dict(color=series_color(cond, label), width=2.4,
                      dash="dot" if label.startswith("scale_matched") else "solid"),
            hovertemplate="%{fullData.name}<br>x = %{x:.3g}<br>share %{y:.5f}"
                          "<extra></extra>"))
        for c, s in zip(centres, share):
            data.append({"run": key, "optimizer": PRETTY[cond], "label": label,
                         "x": c, "share": s})
    add_reference_line(fig, x=1.0, label="x = 1 ULP")
    # Clipped to where the mass is: under 0.01% of coordinates in any run fall
    # outside this window, and the full 13-decade frame leaves half the panel
    # empty.
    fig.update_xaxes(title_text="x = intended update ÷ lattice spacing (log)",
                     type="log", range=[-5, 3], **DECADE_TICKS)
    fig.update_yaxes(title_text="Share of coordinates per bin")
    apply_theme(fig, height=430)
    return fig, data


def verify_bundle(bundles: str, name: str, min_series: int) -> dict:
    """Read the saved render back and confirm the series actually survived.

    A wrong column name drops a trace silently and the file still saves, so the
    check is on the embedded trace JSON rather than on the return value of the
    call that wrote it.
    """
    path = Path(bundles) / name / f"{name}.html"
    html = path.read_text()
    blob = re.search(r"Plotly\.newPlot\(\s*(?:\"[^\"]+\"|'[^']+')\s*,\s*(\[.*)",
                     html, re.S)
    if blob is None:
        raise SystemExit(f"{name}: no Plotly payload in {path}")
    decoder = json.JSONDecoder()
    traces, _ = decoder.raw_decode(blob.group(1))
    real = [t for t in traces if t.get("x") or t.get("y")]
    lengths = [len(t.get("x") or t.get("y") or []) for t in real]
    if len(real) < min_series:
        raise SystemExit(f"{name}: {len(real)} non-empty series, expected "
                         f"at least {min_series}")
    if any(n == 0 for n in lengths):
        raise SystemExit(f"{name}: an embedded series is empty")
    finite = 0
    for t in real:
        for axis in ("x", "y"):
            vals = t.get(axis) or []
            if vals and all(isinstance(v, (int, float)) for v in vals):
                finite += sum(1 for v in vals if math.isfinite(v))
    return {"figure": name, "n_series": len(real), "points": sum(lengths),
            "finite_y": finite,
            "names": [t.get("name") for t in real if t.get("name")]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    root, results = Path(args.root), Path(args.results)
    bundles = str(root / "figures")
    by, curve, hist, mech = load(results)
    write_entity_colors(root, by)

    built = []
    fig, data = fig_by_condition(by)
    save_figure_bundle(fig, "overshoot_by_condition", data=data, root=bundles, alt=(
        "Pooled overshoot with bootstrap intervals for each of the nine runs, "
        "against a reference line at 4/3."))
    built.append("overshoot_by_condition")

    fig, data = fig_runs_vs_magnitude(by)
    save_figure_bundle(fig, "runs_overshoot_vs_magnitude", data=data, root=bundles,
                       alt=("Pooled overshoot against intended update norm on a "
                            "log x axis, one point per training run, coloured by "
                            "update rule, against a reference line at 4/3."))
    built.append("runs_overshoot_vs_magnitude")

    fig, data = fig_shadow_curve(curve)
    save_figure_bundle(fig, "shadow_overshoot_vs_magnitude", data=data, root=bundles, alt=(
        "Pooled overshoot against intended update norm on a log x axis for "
        "AdamW, SGD with momentum, and Muon, measured on one shared trajectory."))
    built.append("shadow_overshoot_vs_magnitude")

    fig, data = fig_shadow_vs_delivery(curve)
    save_figure_bundle(fig, "shadow_overshoot_vs_delivery", data=data, root=bundles,
                       alt=("Pooled overshoot against the fraction of coordinates "
                            "that moved at least one lattice step, on a log x axis, "
                            "for the three update rules on one shared trajectory."))
    built.append("shadow_overshoot_vs_delivery")

    fig, data = fig_shadow_replication(curve)
    if fig is not None:
        save_figure_bundle(fig, "shadow_difference_by_trajectory", data=data,
                           root=bundles, alt=(
            "Paired overshoot difference from AdamW at matched update magnitude, "
            "shown separately for each host trajectory, against a zero line."))
        built.append("shadow_difference_by_trajectory")

    fig, data = fig_step_histogram(hist, by)
    save_figure_bundle(fig, "ulp_step_histogram", data=data, root=bundles, alt=(
        "Share of coordinates moving zero, one, two or more lattice steps, "
        "grouped by optimizer, on a log share axis."))
    built.append("ulp_step_histogram")

    fig, data = fig_mechanism(mech)
    if fig is not None:
        save_figure_bundle(fig, "overshoot_vs_multistep_share", data=data,
                           root=bundles, alt=(
            "Scatter of pooled overshoot against the share of delivered squared "
            "norm carried by moves of two or more lattice steps, for every run "
            "and every shadow magnitude, with the algebraic identity drawn "
            "through them."))
        built.append("overshoot_vs_multistep_share")

    fig, data = fig_x_distribution(hist, by)
    save_figure_bundle(fig, "x_distribution", data=data, root=bundles, alt=(
        "Distribution of the intended update divided by the local lattice "
        "spacing, per optimizer, on a log x axis with a marker at one ULP."))
    built.append("x_distribution")

    expect = {"overshoot_by_condition": 3, "runs_overshoot_vs_magnitude": 3,
              "shadow_overshoot_vs_magnitude": 3,
              "shadow_overshoot_vs_delivery": 3,
              "shadow_difference_by_trajectory": 2, "ulp_step_histogram": 2,
              "overshoot_vs_multistep_share": 3, "x_distribution": 2}
    checks = [verify_bundle(bundles, n, expect[n]) for n in built]
    for c in checks:
        print(f"  checked {c['figure']:34s} {c['n_series']} series, "
              f"{c['points']} points, {c['finite_y']} finite y")
    (root / "figures" / "figure_checks.json").write_text(json.dumps(checks, indent=1))
    print("built bundles: " + ", ".join(built))


if __name__ == "__main__":
    main()
