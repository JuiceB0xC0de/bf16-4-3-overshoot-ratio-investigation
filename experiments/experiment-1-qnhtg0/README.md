# Arm A: is the 4/3 the BF16 writer, or the update distribution?

Deterministic numerical replay of recorded Camera-3 optimizer captures. No
training, no GPU, no new model. The captures store, per detailed coordinate,
the pre-update parameter, both post-update AdamW moments, and the realized
delta, which is enough to reconstruct the *intended* update exactly and then
re-apply that same intent through a different parameter writer.

**Question.** With the intent held byte-identical, does the pooled overshoot
`||r||^2 / <i, r>` change when the parameter is written through a different
dtype or rounding rule?

## Layout

| path | what |
|---|---|
| `src/writers.py` | the five parameter writers and the bf16/fp16 lattice spacings |
| `src/replay.py` | intent reconstruction, pooling, fidelity, lattice geometry |
| `src/bundles.py` | which step bundles are read, and how they are fetched |
| `src/run_arm_a.py` | driver: fetch, verify, replay, write the results JSON |
| `src/analyze.py` | joins the per-step records into plot-ready tables |
| `src/test_writers.py` | writers checked against torch's own bf16/fp16 casts |
| `src/test_replay.py` | pipeline checked against closed forms (4/3 and 12/11) |
| `reference/` | the project's own observer, analyzer, extractor, and published summaries, copied verbatim for provenance |

`replay.py` imports the pooling rule (`_pair`) and the bundle integrity gate
(`verify_dir`) from `reference/transmission_extract.py`, and the control-region
list from `reference/analyze_training_optimizer_delivery_observer.py`, rather
than restating them.

## Method

1. Reconstruct the intended update in float64 with the analyzer's own
   definition: `decay = -(lr*wd)*p_pre`, `adaptive = -(lr/bc1)*m_post /
   (sqrt(v_post)/sqrt(bc2) + eps)`, every scalar read from the fields recorded
   for the owning parameter.
2. Project the pre-update parameter onto each writer's lattice with
   round-to-nearest-even, add the intent in float32 (the precision the observed
   pipeline used), and store the result through the writer.
3. Pool `delivery`, `cosine`, `overshoot`, and the zero fraction over the
   detailed coordinates, and express the same ratio in units of the lattice
   spacing so it can be predicted from the intent alone.

Writers: bf16 round-to-nearest-even, bf16 stochastic, bf16 truncate-toward-zero,
fp16 round-to-nearest-even, fp32.

## Reproduce

```bash
cd src
apy test_writers.py && apy test_replay.py          # closed-form checks
apy run_arm_a.py --out ../results/arm_a.json --bootstrap 300
apy analyze.py --results ../results/arm_a.json --out ../results
```

## Outputs

| output | path | notes |
|---|---|---|
| per-step replay records | `artifact://juiceb0xc0de-15787e/experiments/exp_01m1rbjk2pe84tfg8xj7qnhtg0/arm_a.json` | every writer, subset, histogram, and fidelity number, one record per step bundle |
| joined tables | `results/` | plot-ready CSV/JSON derived from the above |
| report summary | `report_summary.json` | headline claims with their source files |
| claims manifest | `claims_manifest.json` | every quantitative claim mapped to a file and key |

Source data: HuggingFace dataset `juiceb0xc0de/mmb3-slt-6xcam-results` (public),
camera-3 bundles from runs `20260810T091311Z` (187M, clip 1.0),
`20260811T080804Z` (187M, clip 8.0) and `20260814T133327Z` (443M, clip 8.0).
Compute: one CPU-only on-demand job on the `job-core` image. Seed 42.
No external model or LLM credential is used.
