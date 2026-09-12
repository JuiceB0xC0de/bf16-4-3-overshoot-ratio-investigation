"""Write the data-pipeline verification report used as the reference-check gate."""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from data import (MAX_TOKENS, RECORDED_TOKEN_COUNTS, TEMPLATE,  # noqa: E402
                  TOKENIZER_ID, StepData)

d = StepData()
rep = d.verify()
rep.update(d.window_stats(300))
rep["template"] = TEMPLATE
rep["template_rendered_example_row0"] = TEMPLATE.format(
    q=d.rows[0]["question"], a=d.rows[0]["answer"])
rep["recorded_token_counts_checked"] = RECORDED_TOKEN_COUNTS
rep["max_tokens"] = MAX_TOKENS
rep["source_of_truth"] = (
    "token_ids read from camera-3 bundle observed-runs/20260810T091311Z/"
    "e1_s000100 of dataset juiceb0xc0de/mmb3-slt-6xcam-results")
want = np.load(pathlib.Path(__file__).resolve().parent / "step100_token_ids.npy")
got = np.asarray(d.ids_for_step(100))
rep["row99_token_ids_exact_match"] = bool(np.array_equal(got, want))
rep["tokenizer_id"] = TOKENIZER_ID
rep["tokenizer_mirror_note"] = (
    "meta-llama/Llama-3.1-8B is manually gated; the ungated NousResearch "
    "mirror was checked to give byte-identical ids for row 99")

out = pathlib.Path(__file__).resolve().parent.parent / "results" / "data_pipeline.json"
out.write_text(json.dumps(rep, indent=1))
print(json.dumps({k: v for k, v in rep.items()
                  if k != "template_rendered_example_row0"}, indent=1))
print("\nwrote", out)
