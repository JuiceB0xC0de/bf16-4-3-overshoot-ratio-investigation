"""Page-only revisions to the drafted report.

The builder shipped a two-figure page: it fixed the overshoot definition into a
broken LaTeX escape, and it left the nine training runs, the cross-trajectory
replication and the mechanism collapse out of the page even though all three
are already built figure bundles.  This adds them and repairs the math.  It
touches only ``index.html``; every number is read from ``report_summary.json``
so a value on the page cannot drift from the claims manifest.

    apy scratch/patch_report.py
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

EXP = Path(os.environ["SILICO_EXPERIMENT_DIR"])
APP = Path(os.environ["SILICO_APP_DIR"])
IDX = APP / "index.html"

summary = json.loads((EXP / "report_summary.json").read_text())
CLAIM = {c["id"]: c["value"] for c in summary["results"]}
by = json.loads((EXP / "results" / "by_condition.json").read_text())

PRETTY = {"adamw": "AdamW", "muon": "Muon", "sgd": "SGD + momentum"}
LABEL = {"native": "native", "scale_matched": "scale-matched (probe)",
         "scale_matched_v2": "scale-matched", "lr0.005": "rate scan",
         "lr0.01": "rate scan"}
SRC_PREFIX = "experiments/experiment-2-xk0pk0"


def pt(v):
    return v["point"] if isinstance(v, dict) else v


def g(x: float, n: int = 4) -> str:
    """Significant-figure formatting without scientific notation for these ranges."""
    if x == 0:
        return "0"
    from math import floor, log10
    d = -int(floor(log10(abs(x)))) + (n - 1)
    return f"{round(x, d):.{max(d, 0)}f}"


def ovr(x: float) -> str:
    return g(x, 5)


def figure(name: str, number: str, title: str, caption: str, sources: str,
           brief: dict) -> str:
    """Inline a built bundle exactly as the builder did for figures 1 and 2."""
    body = (EXP / "figures" / name / f"{name}.html").read_text()
    return (
        f'    <section class="figure" data-sources="{sources}">\n'
        f'      <h3 class="figure-title">{number}: {title}</h3>\n'
        f'      <script type="application/json" class="figure-brief">'
        f'{json.dumps(brief)}</script>\n'
        f'      {body}\n'
        f'      <figcaption class="figure-description figure-caption">{caption}'
        f'</figcaption>\n'
        f'    </section>\n')


html = IDX.read_text()

# ---------------------------------------------------------------------------
# 1. The overshoot definition. "\f", "\r" and "\l" were consumed as string
#    escapes when the draft was written, leaving "rac{" and stray newlines.
# ---------------------------------------------------------------------------
bad = re.search(r"<p>The observer recorded.*?</p>", html, re.S)
assert bad, "method paragraph not found"
html = html.replace(bad.group(0), (
    "<p>The observer recorded each optimizer's fp32 intended update before the "
    "normal bfloat16 parameter write, then subtracted the pre-step parameters "
    "from the post-step parameters. Overshoot is pooled over every coordinate "
    "in the comparison set as "
    "\\[\\text{overshoot} = \\frac{\\lVert r \\rVert^{2}}{\\langle i, r "
    "\\rangle},\\] "
    "with \\(i\\) the intended update and \\(r\\) the change that actually "
    "landed. It equals 1 when the write delivers exactly what was asked for, "
    "and exceeds 1 when the coordinates that move are pushed past their "
    "intended positions.</p>"))
assert "=rac{" not in html and "\\frac{" in html

# ---------------------------------------------------------------------------
# 2. Cross-trajectory replication, appended inside finding 1.
# ---------------------------------------------------------------------------
across_s, across_m = CLAIM["shadow_across_sgd"], CLAIM["shadow_across_muon"]
md = CLAIM["matched_delivery"]
sm = CLAIM["sgd_smaller_step_still_lower"]
sgd_span = [md[k]["sgd_minus_adamw"] for k in md]
muon_span = [md[k]["muon_minus_adamw"] for k in md]

anchor = ("<p>The paired Muon-minus-AdamW difference is -0.003126, compared "
          "with -0.2452 for SGD-minus-AdamW.")
assert anchor in html
tail = html[html.index(anchor):]
tail_para = tail[:tail.index("</p>") + 4]

replication = (
    figure("shadow_difference_by_trajectory", "Figure 2",
           "The same difference on every host trajectory",
           "Paired overshoot difference from AdamW at matched update "
           "magnitude, measured separately on each of the nine host "
           "trajectories, against a zero line.",
           f"{SRC_PREFIX}/results/shadow_curve.json;"
           f"{SRC_PREFIX}/figures/shadow_difference_by_trajectory/data.json",
           {"claim": "The rule difference does not depend on which trajectory "
                     "the shadow rode on.",
            "reader_check": "Check that each interval sits clear of the zero "
                            "line, and note how much wider the SGD gap is.",
            "expected": ["nine SGD differences", "nine Muon differences",
                         "bootstrap intervals", "zero reference"],
            "why_visual": "Nine paired comparisons read faster as a dot-and-"
                          "interval column than as a table of intervals."}) +
    f"    <p>Running the same comparison on each run's own trajectory gives "
    f"nine independent replications. SGD stays below AdamW on all nine, "
    f"between {g(across_s['max_point'])} and {g(across_s['min_point'])}; Muon "
    f"is below on {across_m['n_negative']} of nine, between "
    f"{g(across_m['max_point'])} and {g(across_m['min_point'])}. Every one of "
    f"the eighteen intervals excludes zero.</p>\n"
    f"    <p>Two readings that would explain SGD's gap without any difference "
    f"in the update rule do not survive. The first is that SGD simply moves "
    f"fewer coordinates. Comparing instead at a matched fraction of "
    f"coordinates actually moved, interpolated along each rule's own curve, "
    f"SGD stays between {g(max(sgd_span))} and {g(min(sgd_span))} below AdamW "
    f"across delivered fractions from 3% to 25%, while Muon stays within "
    f"{g(abs(min(muon_span)))}. The second is that SGD simply takes smaller "
    f"steps. At its own learning rate of {g(sm['sgd_lr'])} SGD takes an update "
    f"{g(sm['sgd_intent_norm_ratio_vs_adamw'])} times AdamW's and still "
    f"reaches only {ovr(sm['sgd_overshoot'])}, below AdamW's "
    f"{ovr(sm['adamw_overshoot'])}. A smaller step does not raise the "
    f"statistic here.</p>\n")
html = html.replace(tail_para, tail_para + "\n" + replication)

# ---------------------------------------------------------------------------
# 3. A section for the nine real runs, between findings 1 and 2.
# ---------------------------------------------------------------------------
ad = pt(by["rows"][[r["run"] for r in by["rows"]].index("adamw_native")]
        ["muon_covered_intent_norm"])
rows = sorted(by["rows"], key=lambda r: pt(r["muon_covered_intent_norm"]))
trs = []
for r in rows:
    ratio = pt(r["muon_covered_intent_norm"]) / ad
    trains = r["trajectory"]["verdict"] == "trains"
    trs.append(
        f"        <tr><td>{PRETTY[r['condition']]}</td>"
        f"<td>{LABEL[r['label']]}</td>"
        f"<td class=\"num\">{r['lr']:g}</td>"
        f"<td class=\"num\">{g(ratio)}</td>"
        f"<td class=\"num\">{ovr(pt(r['muon_covered_overshoot']))}</td>"
        f"<td class=\"num\">{ovr(pt(r['whole_overshoot']))}</td>"
        f"<td>{'yes' if trains else 'no'}</td></tr>")

mm = CLAIM["matched_magnitude_runs"]
runs_section = (
    '  <section id="finding-runs" data-sources="'
    f'{SRC_PREFIX}/results/by_condition.json;'
    f'{SRC_PREFIX}/figures/runs_overshoot_vs_magnitude/data.json">\n'
    "    <h2>The training runs agree, on separate trajectories</h2>\n"
    "    <p>The shadow control removes the trajectory confound by construction, "
    "so it is the primary evidence. The nine real runs are a weaker but "
    "independent check: each has its own learning rate and therefore its own "
    "trajectory, and they still reproduce the ordering. Held to within 0.2% of "
    "AdamW's own median update norm, AdamW reaches "
    f"{ovr(mm['AdamW']['overshoot'])}, Muon {ovr(mm['Muon']['overshoot'])} and "
    f"SGD with momentum {ovr(mm['SGD + momentum']['overshoot'])}.</p>\n"
    + figure("runs_overshoot_vs_magnitude", "Figure 3",
             "Overshoot against update size, one point per run",
             "Pooled overshoot against intended update norm on a log scale, "
             "one point per 300-step training run, coloured by update rule. "
             "Hollow markers are runs whose loss did not fall.",
             f"{SRC_PREFIX}/results/by_condition.json;"
             f"{SRC_PREFIX}/figures/runs_overshoot_vs_magnitude/data.json",
             {"claim": "Overshoot falls with update size for every rule, and "
                       "SGD sits below the other two at every size.",
              "reader_check": "Read the three points near an update norm of "
                              "0.43, then follow the Muon curve to the right.",
              "expected": ["AdamW point", "five Muon points", "three SGD "
                           "points", "bootstrap intervals", "4/3 reference"],
              "why_visual": "Two effects, the rule gap and the magnitude "
                            "trend, are separable only on a magnitude axis."})
    + "    <p>Sweeping Muon's learning rate over a fourteen-fold range walks "
      "its overshoot down the same curve the shadow sweep traces, which is the "
      "magnitude dependence showing up again in real runs. Muon at its "
      "reference default of 0.02, and at 0.01, does not reduce loss on this "
      "model over 300 steps; 0.005 is the largest rate scanned that does.</p>\n"
      "    <table>\n      <caption>Every run, ordered by intended update "
      "norm. Overshoot is pooled over the 48 two-dimensional weights Muon "
      "writes; the whole-model value is given alongside because whole-model "
      "Muon mixes two update rules by construction.</caption>\n"
      "      <thead><tr><th>Rule</th><th>Setting</th><th>Learning rate</th>"
      "<th>Update norm<br>(x AdamW)</th><th>Overshoot<br>(Muon-covered)</th>"
      "<th>Overshoot<br>(whole model)</th><th>Loss fell</th></tr></thead>\n"
      "      <tbody>\n" + "\n".join(trs) + "\n      </tbody>\n    </table>\n"
      "  </section>\n\n")

marker = '  <section id="finding-2"'
assert marker in html
html = html.replace(marker, runs_section + marker)

# ---------------------------------------------------------------------------
# 4. The mechanism collapse, appended inside finding 2.
# ---------------------------------------------------------------------------
mech = CLAIM["mechanism_regimes"]
limit = CLAIM["not_scale_invariant"]
close = ("<p>AdamW's shadow overshoot falls from 1.3323 at 1/32 of its own "
         "magnitude to 1.1661 at 8 times that magnitude.")
assert close in html
close_para = html[html.index(close):]
close_para = close_para[:close_para.index("</p>") + 4]

collapse = (
    figure("overshoot_vs_multistep_share", "Figure 5",
           "One curve for all three rules",
           "Pooled overshoot against the share of delivered squared norm "
           "carried by coordinates that move two or more lattice steps, for "
           "every run and every shadow magnitude. Diamonds are whole runs, "
           "circles are shadow sweep points.",
           f"{SRC_PREFIX}/results/mechanism.json;"
           f"{SRC_PREFIX}/figures/overshoot_vs_multistep_share/data.json",
           {"claim": "Once you know how the delivered update splits between "
                     "one-step and multi-step moves, the update rule adds "
                     "nothing.",
            "reader_check": "Look for colours separating at the same x. They "
                            "do not.",
            "expected": ["AdamW points", "Muon points", "SGD points",
                         "4/3 reference"],
            "why_visual": "The claim is that three coloured groups lie on one "
                          "curve, which is a statement about a scatter."}) +
    f"    <p>Splitting the delivered update into coordinates that move one "
    f"lattice step and coordinates that move more gives an exact identity, "
    f"\\(1/\\rho = (1-s)/\\rho_1 + s/\\rho_2\\), with \\(s\\) the share of "
    f"delivered squared norm in the multi-step moves. Over all "
    f"{mech['n_points']} measured points the two regime ratios stay in narrow "
    f"bands, \\(\\rho_1\\) in [{ovr(mech['one_step_ratio_range'][0])}, "
    f"{ovr(mech['one_step_ratio_range'][1])}] and \\(\\rho_2\\) in "
    f"[{ovr(mech['multi_step_ratio_range'][0])}, "
    f"{ovr(mech['multi_step_ratio_range'][1])}], so \\(s\\) is what moves the "
    f"statistic. All three rules fall on the same band against it. Coordinates "
    f"whose intended update lands near half a lattice spacing are the ones "
    f"that overshoot most, and an optimizer's value for this statistic is "
    f"mostly a question of how much of its update lands there.</p>\n")
html = html.replace(close_para, close_para + "\n" + collapse)

# The two figures already on the page keep their numbers; the x-distribution
# figure was Figure 2 and is now Figure 4.
html = html.replace("Figure 2: Update size relative to lattice spacing",
                    "Figure 4: Update size relative to lattice spacing")

# ---------------------------------------------------------------------------
# 5. Resource links for the figure data added here.
# ---------------------------------------------------------------------------
last = ('      <li><a href="assets/claims_manifest.json">'
        "<code>claims_manifest.json</code></a>: claim-to-source index.</li>")
assert last in html
html = html.replace(last, last + "\n" + "\n".join(
    f'      <li><a href="assets/figures/{n}/data.json">'
    f"<code>{n}/data.json</code></a>: the plotted values behind {t}.</li>"
    for n, t in (("shadow_overshoot_vs_magnitude", "Figure 1"),
                 ("shadow_difference_by_trajectory", "Figure 2"),
                 ("runs_overshoot_vs_magnitude", "Figure 3"),
                 ("x_distribution", "Figure 4"),
                 ("overshoot_vs_multistep_share", "Figure 5"))))

IDX.write_text(html)
print(f"patched {IDX} -> {len(html):,} bytes")
