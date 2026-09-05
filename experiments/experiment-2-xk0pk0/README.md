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

## Design

Six runs of 300 optimizer steps from the same pinned checkpoint over the same
GSM8K rows in the same order: AdamW, SGD with momentum, and Muon, each at its
native learning rate and at a learning rate scale-matched to AdamW's median
intended update norm.

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
| `src/analyze.py` | joins the per-condition records into plot-ready tables |
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

# in the project image, one GPU:
python3 src/run_arm_b.py --mode run --condition adamw --label native \
    --native --lr 2e-4 --steps 300 --wandb          # reference check at step 100
python3 src/run_arm_b.py --mode probe --condition sgd  --lr 0.05  --steps 20
python3 src/run_arm_b.py --mode probe --condition muon --lr 0.02  --steps 20
python3 src/run_arm_b.py --mode run --condition sgd  --label scale_matched --lr <probe> --steps 300
python3 src/run_arm_b.py --mode run --condition muon --label scale_matched --lr <probe> --steps 300
python3 src/analyze.py --out results
```

## Outputs

| output | path | notes |
|---|---|---|
| per-condition step records | `$SILICO_EXPERIMENT_ARTIFACTS_DIR/armb_<condition>_<label>.json` | every step's pooled stats, geometry, and full captures |
| plot-ready rows | same, `.rows.jsonl` | one flat row per step |
| joined tables | `results/` | written by `src/analyze.py` |
| data pipeline gate | `results/data_pipeline.json` | template, tokenizer, exact-match check |

Model: `state-spaces/mamba3-siso-187m` @ `6792c27c00f3bb41506db1066dcd1c51bb0f4b02`.
Data: `openai/gsm8k` `main/train`, rows in order, batch 1, max 512 tokens.
Image: `docker.io/juiceboxdocks/mamba3-atlas:1cde57a2d4981702add8c542ec75c7b2282b2ce3`,
the image recorded in every capture manifest of the source dataset.
Seed 1234. Recorded reference values from `reference/capture_summary_clip1.jsonl`.
