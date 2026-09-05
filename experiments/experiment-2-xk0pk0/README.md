# Arm B: is 4/3 an AdamW number, or does any optimizer get it?

The pooled overshoot `||r||^2 / <i, r>` on this project's 187M Mamba-3 runs sits
near 4/3 and is described as AdamW's gradient delivery degrading. Nothing
measured so far separates AdamW from the bfloat16 writer it feeds. Arm A held
the intent fixed and swapped the writer. This arm does the complement: it holds
the model, the data, the step count and the bf16 write path fixed, generates
genuinely different update distributions from three optimizers, and asks
whether the overshoot follows the optimizer or stays put.

**Question.** Does the pooled overshoot change when a different optimizer
produces the intended update?

**Answer.** Both framings are wrong. Muon reproduces AdamW's overshoot to
within 0.0031 at matched update magnitude on a shared trajectory (1.3208 vs
1.3241), so 4/3 is not an AdamW number. SGD with momentum sits 0.2452 below it
under identical gradients and identical bfloat16 parameters (1.0792), so it is
not a property of the bfloat16 writer alone either. The real runs agree:
held to AdamW's own update magnitude to within 0.2%, AdamW reaches 1.3246,
Muon 1.3200 and SGD 1.1702.

What the statistic reads out is where the update sits relative to the local
lattice spacing. Writing a bfloat16 parameter rounds to nearest, so a
coordinate whose intended update is `x` lattice spacings moves `round(x)`
spacings; predicting each run's overshoot from its measured distribution of
`x` alone reproduces the measurement (median gap 0.0013, worst 0.017 over nine
runs), leaving nothing optimizer-specific. 4/3 is exactly what that prediction
returns when `x` is uniform on the unit interval, which is the small-update
limit rather than a constant of the writer. See `results/` and the published
report.

## Design

300-step runs from the same pinned checkpoint over the same GSM8K rows in the
same order: AdamW, SGD with momentum, and Muon, each at its native learning
rate and at a learning rate scale-matched to AdamW's median intended update
norm. AdamW's scale-matched cell is the reference condition itself, so the
planned matrix is five runs, plus a Muon learning-rate scan and a corrected
scale-match added after the first results (below).

**Shadow control.** Matching magnitude through a learning rate puts each
optimizer on its own trajectory, so a difference between rows confounds the
update rule with the model state the run reached. Every run therefore also
carries a read-only shadow: on that run's own trajectory, each of the three
rules is fed the identical post-clip gradients and its intent is written into
the identical bfloat16 parameters, at nine magnitudes from 1/32 to 8 times the
host's own median. The shadow matches magnitude exactly and shares the
trajectory, and it is self-validating because the shadow's reconstruction of
the host optimizer must reproduce the host's measured overshoot. The two
measurements answer different questions and are reported separately.

**Two corrections the first results forced.** Muon at its reference default of
0.02 diverges on this model over 300 steps (loss 2.504 to 3.379), which a
20-step probe could not see, so its overshoot there is measured on a diverging
trajectory; a 300-step learning-rate scan found that 0.01 does not train either
and 0.005 does, which is the largest Muon rate here that reduces loss.
The scale-matched multipliers, also fixed from the 20-step probe, came in at
1.649 (Muon) and 1.537 (SGD) times AdamW's 300-step median rather than 1.0, so
both were rerun with corrected multipliers and landed at 0.9989 and 0.9983.
The original runs are kept as extra points on the magnitude curve, which turns
the nine runs into a learning-rate ladder spanning a 48-fold range of update
norm.

Muon only orthogonalizes 2-D hidden weights, so the primary comparison is
restricted to the parameters Muon actually writes, with the whole-model number
reported alongside. All three conditions apply the identical pair of in-place
writes per step, `p.mul_(1 - lr*wd)` then one additive write, so the write path
is a constant rather than a co-varying factor.

**Reference check.** The AdamW run must reproduce the recorded step-100
behaviour of run `20260810T091311Z` (overshoot within 0.02 of 1.325717, bf16
zero fraction within 0.02 of 0.938561, clipping firing on every step). If it
does not, the instrument is wrong and the SGD and Muon numbers mean nothing.

## Layout

| path | what |
|---|---|
| `src/data.py` | GSM8K row -> the exact token sequence the recorded run used |
| `src/geometry.py` | pooled delivery/cosine/overshoot and bf16 lattice geometry, accumulated per parameter |
| `src/conditions.py` | model load, parameter roster, the three optimizer conditions and their intent rules |
| `src/vendor_muon.py` | Muon, vendored from KellerJordan/Muon `master` |
| `src/observer.py` | read-only per-step capture of intent, realization and lattice geometry |
| `src/run_arm_b.py` | driver: `verify`, `probe`, `run` |
| `src/shadow.py` | the read-only shadow optimizers, magnitude sweep, and self-check |
| `src/analyze.py` | joins the per-condition records into plot-ready tables |
| `src/figures.py` | builds and verifies the report figure bundles |
| `src/preflight_report.py` | 12-stage batch diagnostic run before the first production job |
| `src/test_geometry.py` | the torch geometry port checked against Arm A and closed forms |
| `reference/` | the project's own observer, analyzer, extractor and published summaries, plus Arm A's writers/replay, copied for provenance |
| `results/` | small reviewable evidence |

## Provenance of the data pipeline

The original trainer was not supplied, so the prompt format was read off the
recorded captures rather than guessed: camera-3 bundles store `token_ids`, and
decoding the step-100 bundle of run `20260810T091311Z` gives

```
<|begin_of_text|>Question:\n{question}\n\nAnswer:\n{answer}<|end_of_text|>
```

with loss over every position. Re-encoding row 99 reproduces the recorded 230
token ids exactly, and `len(ids) - 1` matches the recorded
`supervised_token_count` on all eight distinct rows in the published
transmission table. See `results/data_pipeline.json`.

## How to reproduce

```bash
apy src/test_geometry.py                      # geometry port vs Arm A, CPU, seconds
apy scratch/emit_data_report.py               # data pipeline gate -> results/data_pipeline.json

# in the project image, one GPU per job.  --shadow is what makes the run
# carry the confound-free control; --native turns on the reference check.
python3 src/run_arm_b.py --mode run --condition adamw --label native \
    --native --lr 2e-4     --steps 300 --shadow --wandb
python3 src/run_arm_b.py --mode run --condition sgd  --label native \
    --native --lr 0.01      --steps 300 --shadow --wandb
python3 src/run_arm_b.py --mode run --condition sgd  --label scale_matched_v2 \
             --lr 0.037930  --steps 300 --shadow --wandb
python3 src/run_arm_b.py --mode run --condition muon --label scale_matched_v2 \
             --lr 0.0014736 --steps 300 --shadow --wandb
python3 src/run_arm_b.py --mode run --condition muon --label lr0.005 \
             --lr 0.005     --steps 300 --shadow --wandb

# pod-side, from the downloaded records:
uv run --no-sync python src/analyze.py --inputs <dir of armb_*.json> --out results
uv run --no-sync python src/figures.py --results results --root .
```

## Outputs

| output | path | notes |
|---|---|---|
| per-condition step records | `artifact://juiceb0xc0de-15787e/experiments/exp_01m1rc45nwevdtjx8af5xk0pk0/armb_<condition>_<label>.json` | every step's pooled stats, lattice geometry, shadow sweep, and full captures; about 5.3 MB each |
| plot-ready rows | same prefix, `.rows.jsonl` | one flat row per step |
| job logs | same prefix, `log_<name>.txt` | full stdout of each production job |
| batch preflight | same prefix, `preflight_adamw.json` | 12-stage diagnostic, all passed |
| joined tables | `results/by_condition.json`, `shadow_curve.json`, `step_histograms.json`, `mechanism.json`, `by_condition.csv` | written by `src/analyze.py` |
| report contract | `report_summary.json`, `claims_manifest.json` | written by `src/make_report_summary.py`; every published number traces to a file and key |
| figure bundles | `figures/<name>/` | interactive render, plot-ready `data.json`, alt text; `figures/figure_checks.json` records the read-back verification |
| data pipeline gate | `results/data_pipeline.json` | template, tokenizer, exact-match check |

No external model API was used: every number comes from the pinned checkpoint
running locally in the project image. W&B project
`mamba3-arm-b-optimizer-overshoot` carries the per-step curves.

Model: `state-spaces/mamba3-siso-187m` @ `6792c27c00f3bb41506db1066dcd1c51bb0f4b02`.
Data: `openai/gsm8k` `main/train`, rows in order, batch 1, max 512 tokens.
Image: `docker.io/juiceboxdocks/mamba3-atlas:1cde57a2d4981702add8c542ec75c7b2282b2ce3`,
the image recorded in every capture manifest of the source dataset.
Seed 1234. Recorded reference values from `reference/capture_summary_clip1.jsonl`.
