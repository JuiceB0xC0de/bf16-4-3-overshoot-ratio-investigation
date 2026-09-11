"""Render the Learning Transmission Atlas from a developed table.

Two charts, because the two questions are different.

**Transmission heatmap** screens a run.  Rows are metrics grouped by stage and
by the transitions between stages; columns are observed steps; colour is the
within-run robust percentile so quantities with incompatible units can share a
panel.  Raw values are printed in the cells, because a percentile finds an
event and only the raw number can be published.

**Fidelity/efficacy portrait** is the scientific claim.  Geometric fidelity
between what the optimizer intended and what the parameters actually stored is
one axis; functional benefit is the other.  If those two were the same thing
the cloud would lie on a rising diagonal.  Whether it does is the entire
question, and this is the plot that can answer it.

A third panel is drawn because the data forced it: **overshoot**.  For ideal
``i`` and realized ``r``,

    delivery = |r|/|i|,  cosine = <i,r>/(|i||r|),  overshoot = delivery/cosine

Pure support truncation gives ``r = P i``, hence ``cosine == delivery`` and
overshoot exactly 1.0.  Measured, it sits near 1.34 and barely moves across
thousands of steps, clipped and unclipped alike.  So realization is not a mask:
the coordinates that survive are displaced outward, which is the signature of
round-to-nearest on the parameter's own bf16 lattice.

    python transmission_atlas.py transmission.json -o atlas
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# Rows of the heatmap, in pipeline order.  ``higher_is_more`` only controls the
# colour direction, never the printed value.
PANELS = [
    ("OPTIMIZER INTENT", [
        ("grad_norm_preclip", "gradient norm (pre-clip)", True),
        ("clip_coefficient", "clip coefficient", False),
        ("ideal_norm", "intended update norm", True),
        ("innovation_over_carry", "innovation / carried history", True),
    ]),
    ("INTENT -> REALIZATION", [
        ("delivery", "norm delivery  |r|/|i|", True),
        ("cosine", "direction fidelity  cos(i,r)", True),
        ("overshoot", "OVERSHOOT  delivery/cosine", True),
        ("zero_fraction_bf16", "zero-realized fraction (bf16)", True),
        ("zero_fraction_fp32", "zero-realized fraction (fp32)", True),
        ("embedding_delta_share", "embedding share of write", True),
    ]),
    ("REPRESENTATION", [
        ("peak_mixer_movement", "peak mixer movement", True),
        ("peak_mixer_layer", "peak layer", True),
        ("final_boundary_movement", "final boundary movement", True),
    ]),
    ("FUNCTION", [
        ("loss_pre", "pre-update loss (own batch)", True),
        ("efficacy", "efficacy (loss improvement)", True),
    ]),
]


def robust_percentile(values: np.ndarray) -> np.ndarray:
    """Rank within the run, so unit-incompatible rows can share a colour map."""
    finite = np.isfinite(values)
    out = np.full(values.shape, np.nan)
    if finite.sum() <= 1:
        out[finite] = 0.5
        return out
    ranks = values[finite].argsort().argsort().astype(float)
    out[finite] = ranks / max(len(ranks) - 1, 1)
    return out


def fmt(value: float, metric: str) -> str:
    if not np.isfinite(value):
        return ""
    if metric == "peak_mixer_layer":
        return f"L{int(value)}"
    if abs(value) >= 100 or (abs(value) < 0.001 and value != 0):
        return f"{value:.1e}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}"


def draw_heatmap(steps, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    labels = [str(s["optimizer_step"]) for s in steps]
    rows, row_labels, seps, headers = [], [], [], []
    index = 0
    for panel, metrics in PANELS:
        headers.append((index, panel))
        for metric, label, higher in metrics:
            values = np.array(
                [float(s.get(metric, np.nan)) for s in steps], dtype=float
            )
            pct = robust_percentile(values)
            if not higher:
                pct = 1.0 - pct
            rows.append((pct, values, metric))
            row_labels.append(label)
            index += 1
        seps.append(index)

    cmap = LinearSegmentedColormap.from_list(
        "transmission",
        ["#10141c", "#1d3557", "#2a9d8f", "#e9c46a", "#e76f51"],
    )
    grid = np.array([r[0] for r in rows])

    height = 0.42 * len(rows) + 2.4
    fig, ax = plt.subplots(figsize=(1.05 * len(steps) + 6.5, height))
    fig.patch.set_facecolor("#0b0e14")
    ax.set_facecolor("#0b0e14")
    ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=1,
              interpolation="nearest")

    for i, (pct, values, metric) in enumerate(rows):
        for j, value in enumerate(values):
            shade = pct[j]
            colour = "#0b0e14" if np.isfinite(shade) and shade > 0.55 else "#d8dee9"
            ax.text(j, i, fmt(value, metric), ha="center", va="center",
                    fontsize=7.2, color=colour)

    for y in seps[:-1]:
        ax.axhline(y - 0.5, color="#8892a4", lw=1.4)
    for j, step in enumerate(steps):
        if step.get("clip_fired"):
            ax.axvline(j, color="#e76f51", lw=1.0, alpha=0.35)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8.5,
                       color="#d8dee9")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8.5, color="#d8dee9")
    for spine in ax.spines.values():
        spine.set_color("#333a48")
    ax.tick_params(colors="#8892a4", length=0)

    for y, panel in headers:
        ax.text(-0.9, y - 0.62, panel, fontsize=8, color="#7fd1c1",
                fontweight="bold", ha="right", va="center")

    contaminated = any(
        s.get("efficacy_source", "").startswith("same_batch") for s in steps
    )
    note = (
        "efficacy axis = same-batch loss (example-difficulty confound present)"
        if contaminated
        else "efficacy axis = held-out probe"
    )
    ax.set_title(
        "Learning Transmission Heatmap\n"
        "colour = within-run percentile, text = raw value, "
        f"red rule = clipping fired\n{note}",
        fontsize=11, color="#d8dee9", pad=16,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def draw_portrait(steps, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cos = np.array([s["cosine"] for s in steps])
    eff = np.array([s["efficacy"] * 100 for s in steps])
    move = np.array([s["final_boundary_movement"] for s in steps])
    deliver = np.array([s["delivery"] for s in steps])
    clipped = np.array([bool(s["clip_fired"]) for s in steps])
    over = np.array([s["overshoot"] for s in steps])
    label = np.array([s["optimizer_step"] for s in steps])

    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(15.5, 7.0), gridspec_kw={"width_ratios": [1.15, 1]}
    )
    fig.patch.set_facecolor("#0b0e14")

    for a in (ax, bx):
        a.set_facecolor("#0b0e14")
        a.tick_params(colors="#8892a4")
        for spine in a.spines.values():
            spine.set_color("#333a48")
        a.grid(alpha=0.12, color="#8892a4", lw=0.6)

    sizes = 90 + 3400 * (move - move.min()) / max(float(np.ptp(move)), 1e-12)
    scatter = ax.scatter(
        cos, eff, s=sizes, c=deliver, cmap="viridis",
        edgecolors=np.where(clipped, "#e76f51", "#8892a4"),
        linewidths=np.where(clipped, 2.4, 1.0), alpha=0.88, zorder=3,
    )
    for x, y, name in zip(cos, eff, label):
        ax.annotate(str(name), (x, y), fontsize=7.5, color="#d8dee9",
                    xytext=(0, -3), textcoords="offset points",
                    ha="center", va="center", zorder=4)

    if len(cos) > 2:
        r = float(np.corrcoef(cos, eff)[0, 1])
        ax.plot(
            np.sort(cos),
            np.poly1d(np.polyfit(cos, eff, 1))(np.sort(cos)),
            color="#e9c46a", lw=1.2, ls="--", alpha=0.75, zorder=2,
            label=f"least squares, r = {r:+.3f}",
        )
        ax.legend(fontsize=8.5, facecolor="#141924", edgecolor="#333a48",
                  labelcolor="#d8dee9")

    ax.set_xlabel("geometric fidelity   cos(intended, realized)", fontsize=10,
                  color="#d8dee9")
    ax.set_ylabel("functional efficacy   loss improvement (%)", fontsize=10,
                  color="#d8dee9")
    ax.set_title(
        "Fidelity does not buy efficacy\n"
        "size = final-boundary movement, colour = norm delivery, "
        "orange ring = clipped",
        fontsize=11, color="#d8dee9", pad=12,
    )
    bar = fig.colorbar(scatter, ax=ax, pad=0.015)
    bar.set_label("norm delivery  |realized| / |intended|", color="#d8dee9",
                  fontsize=9)
    bar.ax.tick_params(colors="#8892a4")
    bar.outline.set_edgecolor("#333a48")

    # Right panel: the truncation identity, and how far off it the data sits.
    lo = min(cos.min(), deliver.min()) * 0.94
    hi = max(cos.max(), deliver.max()) * 1.06
    bx.plot([lo, hi], [lo, hi], color="#8892a4", lw=1.4, ls="--",
            label="pure support truncation:  cos = delivery")
    bx.scatter(deliver, cos, s=170, c=over, cmap="magma",
               edgecolors=np.where(clipped, "#e76f51", "#8892a4"),
               linewidths=np.where(clipped, 2.2, 1.0), zorder=3)
    for x, y, name in zip(deliver, cos, label):
        bx.annotate(str(name), (x, y), fontsize=7, color="#d8dee9",
                    xytext=(7, -2), textcoords="offset points")
    for x, y in zip(deliver, cos):
        bx.plot([x, x], [y, x], color="#e76f51", lw=0.9, alpha=0.5, zorder=2)

    bx.set_xlim(lo, hi)
    bx.set_ylim(lo, hi)
    bx.set_xlabel("norm delivery  |r| / |i|", fontsize=10, color="#d8dee9")
    bx.set_ylabel("direction fidelity  cos(i, r)", fontsize=10, color="#d8dee9")
    bx.set_title(
        f"Realization is not a mask\n"
        f"overshoot = delivery/cosine, median {np.median(over):.4f} "
        f"(1.000 would be a pure mask)",
        fontsize=11, color="#d8dee9", pad=12,
    )
    bx.legend(fontsize=8.5, facecolor="#141924", edgecolor="#333a48",
              labelcolor="#d8dee9", loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("table", type=Path)
    parser.add_argument("-o", "--out", type=Path, default=Path("atlas"))
    args = parser.parse_args()

    payload = json.loads(args.table.read_text())
    steps = payload["steps"]
    args.out.mkdir(parents=True, exist_ok=True)

    heatmap = args.out / "transmission-heatmap.png"
    portrait = args.out / "fidelity-efficacy-portrait.png"
    draw_heatmap(steps, heatmap)
    draw_portrait(steps, portrait)

    over = np.array([s["overshoot"] for s in steps])
    cos = np.array([s["cosine"] for s in steps])
    eff = np.array([s["efficacy"] for s in steps])
    print(f"wrote {heatmap}")
    print(f"wrote {portrait}")
    print()
    print(f"steps            {len(steps)}")
    print(f"overshoot        median {np.median(over):.4f}  "
          f"range {over.min():.4f} to {over.max():.4f}")
    print(f"fidelity/efficacy correlation  r = {np.corrcoef(cos, eff)[0, 1]:+.4f}")
    if any(s.get("efficacy_source", "").startswith("same_batch") for s in steps):
        print("efficacy is same-batch loss: this correlation cannot separate "
              "learning from example difficulty.")


if __name__ == "__main__":
    main()
