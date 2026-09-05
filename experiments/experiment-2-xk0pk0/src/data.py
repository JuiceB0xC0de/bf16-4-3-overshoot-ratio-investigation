"""GSM8K rows -> the exact token sequence the recorded run put through the model.

The template is not a guess.  The recorded camera-3 bundles store ``token_ids``
for every capture, so the format was read straight off the step-100 bundle of
run ``20260810T091311Z`` and then checked two ways:

* the reconstructed ids for row 99 are byte-identical to the recorded array
  (230 ids, sha-checked in ``verify()``), and
* ``len(ids) - 1`` matches the recorded ``supervised_token_count`` on all eight
  distinct rows that appear in the published transmission table, which span
  rows 655-7387.

Format::

    <|begin_of_text|>Question:\\n{question}\\n\\nAnswer:\\n{answer}<|end_of_text|>

Loss is over every position: the recorded ``supervised_token_count`` equals
``len(token_ids) - 1``, i.e. targets are ``ids[1:]`` with no prompt masking.

The tokenizer is Llama-3.1-8B.  ``meta-llama/Llama-3.1-8B`` is manually gated,
so the default here is the ungated ``NousResearch`` mirror; both were checked
to produce byte-identical ids for row 99, so the mirror is not a variable.
"""

from __future__ import annotations

import numpy as np

TEMPLATE = "Question:\n{q}\n\nAnswer:\n{a}"
TOKENIZER_ID = "NousResearch/Meta-Llama-3.1-8B"
GSM8K_REPO = "openai/gsm8k"
GSM8K_FILE = "main/train-00000-of-00001.parquet"
MAX_TOKENS = 512

# Recorded ``supervised_token_count`` == len(ids) - 1, from the published
# transmission table.  Used as an integrity check, never to select a template.
RECORDED_TOKEN_COUNTS = {
    655: 127, 656: 85, 657: 168, 658: 192,
    1679: 118, 1680: 72, 1681: 133, 7387: 161,
}
# Row 99 is the reference-check step (optimizer step 100).
RECORDED_ROW99_LEN = 230


def load_rows() -> list:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(GSM8K_REPO, GSM8K_FILE, repo_type="dataset")
    return pq.read_table(path).to_pylist()


def load_tokenizer(tokenizer_id: str = TOKENIZER_ID):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_id)


def encode_row(tokenizer, row: dict) -> list:
    """One dataset row -> the full token id sequence, BOS/EOS included."""
    body = TEMPLATE.format(q=row["question"], a=row["answer"])
    ids = tokenizer(body, add_special_tokens=False)["input_ids"]
    ids = [tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id]
    return ids[:MAX_TOKENS]


class StepData:
    """Deterministic step -> token ids.  Step k consumes dataset row k-1."""

    def __init__(self, tokenizer_id: str = TOKENIZER_ID):
        self.rows = load_rows()
        self.tokenizer = load_tokenizer(tokenizer_id)
        self.tokenizer_id = tokenizer_id
        self.epoch_len = len(self.rows)

    def row_index(self, step: int) -> int:
        return (step - 1) % self.epoch_len

    def ids_for_step(self, step: int) -> list:
        return encode_row(self.tokenizer, self.rows[self.row_index(step)])

    def verify(self) -> dict:
        """Integrity gate: template and tokenizer must reproduce the record."""
        row99 = self.ids_for_step(100)
        counts = {
            r: len(encode_row(self.tokenizer, self.rows[r])) - 1
            for r in RECORDED_TOKEN_COUNTS
        }
        mismatched = {
            r: (counts[r], RECORDED_TOKEN_COUNTS[r])
            for r in RECORDED_TOKEN_COUNTS
            if counts[r] != RECORDED_TOKEN_COUNTS[r]
        }
        report = {
            "tokenizer_id": self.tokenizer_id,
            "epoch_len": self.epoch_len,
            "row99_len": len(row99),
            "row99_len_expected": RECORDED_ROW99_LEN,
            "row99_len_ok": len(row99) == RECORDED_ROW99_LEN,
            "recorded_counts_ok": not mismatched,
            "mismatched": mismatched,
            "row99_first8": row99[:8],
            "row99_last4": row99[-4:],
        }
        if not (report["row99_len_ok"] and report["recorded_counts_ok"]):
            raise AssertionError(f"data pipeline does not reproduce the record: {report}")
        return report

    def window_stats(self, n_steps: int) -> dict:
        lens = [len(self.ids_for_step(s)) for s in range(1, n_steps + 1)]
        arr = np.asarray(lens)
        return {
            "n_steps": n_steps,
            "len_min": int(arr.min()),
            "len_median": float(np.median(arr)),
            "len_max": int(arr.max()),
            "n_truncated_at_max_tokens": int((arr >= MAX_TOKENS).sum()),
            "total_supervised_tokens": int((arr - 1).sum()),
        }
