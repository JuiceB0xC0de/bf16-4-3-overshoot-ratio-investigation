"""Second patch pass over the published report page.

Applies the four review corrections:

f1  scope the -0.2452 SGD deficit to the shadow's decay-free SGD rule, give the
    real decayed number, and attribute the shadow-vs-real SGD gap to the
    omitted decay rather than to trajectory;
f2  restate the shadow self-validation per host;
f3  present the predicted-from-x agreement as a check of the round-to-nearest
    write model and complete mediation by x, and scope 4/3 to broadly spread
    update distributions;
f4  drop the sign-replication wording for Muon and disclose that SGD's deficit
    size tracks model state.

Every replacement is exact-match and asserted, so a silent no-op cannot ship.
"""

from __future__ import annotations

import os
import pathlib

PAGE = pathlib.Path(os.environ["SILICO_APP_DIR"]) / "index.html"

EDITS: list[tuple[str, str]] = [
    # ---- f1: takeaway 1 -------------------------------------------------
    (
        "At AdamW's update magnitude, the shadow control gives AdamW 1.3241, "
        "Muon 1.3208, and SGD with momentum 1.0792.",

        "At AdamW's update magnitude, the shadow control gives AdamW 1.3241, "
        "Muon 1.3208, and a decay-free SGD with momentum 1.0792. The real SGD "
        "run, which carries its weight decay, sits 0.1544 below AdamW.",
    ),
    # ---- f3: takeaway 3 -------------------------------------------------
    (
        "<li><strong>Four-thirds is the small-update limit of round-to-nearest "
        "writing, not a constant.</strong> A prediction from each run's "
        "update-to-lattice-spacing distribution alone reproduces all nine "
        "measurements, and the measured value falls as updates grow.",

        "<li><strong>Four-thirds is what round-to-nearest writing gives for "
        "small updates spread broadly across the lattice, not a "
        "constant.</strong> Under that write model, each run's "
        "update-to-lattice-spacing distribution alone accounts for its measured "
        "value, and the value falls as updates grow.",
    ),
    # ---- f1: section lede -----------------------------------------------
    (
        "<p>Muon nearly matches AdamW at the same magnitude, while SGD remains "
        "lower by 0.2452.",

        "<p>Muon nearly matches AdamW at the same magnitude, while SGD remains "
        "lower by 0.2452 in the control's decay-free form and by 0.1544 in the "
        "real decayed run.",
    ),
    # ---- f1: paragraph under Figure 1 -----------------------------------
    (
        "<p>The paired Muon-minus-AdamW difference is -0.003126, compared with "
        "-0.2452 for SGD-minus-AdamW.",

        "<p>The paired Muon-minus-AdamW difference is -0.003126, compared with "
        "-0.2452 for SGD-minus-AdamW. The shadow applies every rule without the "
        "decoupled weight-decay write, so that -0.2452 belongs to a decay-free "
        "SGD; the real decayed comparison is below.",
    ),
    # ---- f4: figure 2 title and brief -----------------------------------
    (
        "<h3 class=\"figure-title\">Figure 2: The same difference on every host "
        "trajectory</h3>",

        "<h3 class=\"figure-title\">Figure 2: The difference from AdamW on "
        "every host trajectory</h3>",
    ),
    (
        "{\"claim\": \"The rule difference does not depend on which trajectory "
        "the shadow rode on.\", \"reader_check\": \"Check that each interval "
        "sits clear of the zero line, and note how much wider the SGD gap "
        "is.\"",

        "{\"claim\": \"SGD is below AdamW on every trajectory and Muon is "
        "within 0.014 of it on every trajectory, though the size of the SGD "
        "deficit varies with model state.\", \"reader_check\": \"Check that "
        "each interval sits clear of the zero line, note how much wider the SGD "
        "gap is, and compare the two diverged Muon hosts with the other "
        "seven.\"",
    ),
    # ---- f4: paragraph under Figure 2 -----------------------------------
    (
        "<p>Running the same comparison on each run's own trajectory gives nine "
        "independent replications. SGD is below AdamW on all nine, by 0.06653 "
        "to 0.2481; Muon is below on eight of the nine, ranging from -0.01394 "
        "to +0.003136. Every one of the eighteen intervals excludes zero.</p>",

        "<p>Running the same comparison on each run's own trajectory gives nine "
        "independent replications, and every one of the eighteen intervals "
        "excludes zero. SGD is below AdamW on all nine, but by an amount that "
        "tracks model state: 0.2113 to 0.2481 on the seven hosts whose loss "
        "fell, and only 0.1444 and 0.06653 on the two diverged Muon hosts. For "
        "Muon what replicates is the size of the difference rather than its "
        "sign: it stays within 0.01394 of AdamW everywhere, negative on eight "
        "hosts and +0.003136 on the diverged Muon-at-0.02 host.</p>\n"
        "    <p>The control's SGD rule is decay-free, and at SGD's learning "
        "rates that matters. Replaying a host's own rule through the shadow "
        "reproduces that host's intended-update norm to six figures, so the "
        "decay write is the only thing left that differs between the two. On "
        "the AdamW host and the five Muon hosts, where the decay write is at "
        "most 0.051 lattice spacings, the reconstruction lands within 0.003978 "
        "of the measured value. On the three SGD hosts, where it reaches "
        "0.09672 and 0.1487, the decay-free reconstruction sits 0.05431 to "
        "0.1186 low. Restoring the decay is what closes that gap, not a change "
        "of trajectory: at 0.9983 of AdamW's own update norm the real SGD run "
        "reads 1.1702 against AdamW's 1.3246, a deficit of 0.1544. SGD is well "
        "below AdamW either way, so the conclusion does not turn on which "
        "version is quoted.</p>",
    ),
    # ---- f3: mechanism paragraph ----------------------------------------
    (
        "Applying that rule to each measured \\(x\\) distribution predicts the "
        "nine run-level overshoots with a median absolute gap of 0.001320 and a "
        "worst gap of 0.01730.</p>",

        "Applying that rule to each measured \\(x\\) distribution recovers the "
        "nine run-level overshoots with a median absolute gap of 0.001320 and a "
        "worst gap of 0.01730. The realized write is by construction a "
        "round-to-nearest write of the same intent, so this is a check of that "
        "write model and a measure of its residuals rather than independent "
        "evidence across optimizers. What it establishes is that, given the "
        "model, \\(x\\) mediates the optimizer's effect completely: no term for "
        "which rule produced the update is needed. The two largest residuals "
        "are the two runs with the largest decay write, which the prediction "
        "does not carry.</p>",
    ),
    # ---- f3: the 4/3 sentence -------------------------------------------
    (
        "The round-to-nearest prediction returns exactly 4/3 when \\(x\\) is "
        "uniform on \\([0,1]\\), which identifies 4/3 as a small-update limit "
        "rather than a writer constant.</p>",

        "The round-to-nearest prediction returns exactly 4/3 when \\(x\\) is "
        "uniform on \\([0,1]\\), so 4/3 is the small-update limit for a rule "
        "whose \\(x\\) is spread broadly enough to look locally uniform. "
        "AdamW's and Muon's are: their 5th to 95th percentiles of \\(x\\) span "
        "factors of 179 and 201, and their curves flatten at 1.3323 and 1.3331. "
        "SGD's spans a factor of 2.6, and its curve tops out at 1.2368 at the "
        "smallest magnitude swept, never approaching 4/3 anywhere in the "
        "256-fold sweep. The limit belongs to the distribution, not to the "
        "writer and not to any one optimizer.</p>",
    ),
    # ---- f2: method self-validation --------------------------------------
    (
        "The shadow reconstruction also tracked the host optimizer on all 300 "
        "steps, with a worst absolute overshoot gap of 0.002811.</p>",

        "The shadow's AdamW reconstruction tracked the real AdamW step on all "
        "300 steps of its own host, with a worst absolute overshoot gap of "
        "0.002811. That stored check always compares the shadow's AdamW rule "
        "against the host, so it is a reconstruction test only on that row, and "
        "it is recorded as failing on the other eight, where it compares "
        "mismatched rules. The per-host test is instead the host's own rule "
        "read off the shadow curve at matched magnitude: within 0.003978 on the "
        "AdamW host and the five Muon hosts, and 0.05431 to 0.1186 low on the "
        "three SGD hosts, which is the weight decay the shadow omits.</p>",
    ),
    # ---- f1: method design line ------------------------------------------
    (
        "<dd>Three shadow rules at nine exact magnitudes on every host "
        "trajectory.</dd>",

        "<dd>Three shadow rules, without the decoupled decay write, at nine "
        "exact magnitudes on every host trajectory.</dd>",
    ),
    # ---- f1 + f3: limitations --------------------------------------------
    (
        "<li>This covers one model, one size, GSM8K, and 300 steps.",

        "<li>The shadow omits the decoupled weight-decay write, which is a "
        "thousand times below the rounding threshold at AdamW's learning rate "
        "but not at SGD's, so its SGD numbers describe a decay-free SGD. The "
        "real decayed runs put SGD 0.1544 below AdamW at matched magnitude "
        "rather than 0.2452; both are well below. Four-thirds is a limit for "
        "broadly spread update distributions, which AdamW and Muon have here "
        "and SGD does not.</li>\n"
        "      <li>This covers one model, one size, GSM8K, and 300 steps.",
    ),
]


def main() -> None:
    text = PAGE.read_text()
    for i, (old, new) in enumerate(EDITS):
        count = text.count(old)
        assert count == 1, f"edit {i}: matched {count} times, not 1\n{old[:110]!r}"
        text = text.replace(old, new)
    PAGE.write_text(text)
    print(f"applied {len(EDITS)} edits to {PAGE} ({len(text)} bytes)")


if __name__ == "__main__":
    main()
