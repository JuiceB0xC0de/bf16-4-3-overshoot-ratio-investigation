"""Identify the GSM8K -> token_ids reconstruction from recorded (row, n_tokens).

n_tokens == supervised_token_count == len(token_ids) - 1, so a candidate is
accepted only if len(tokenize(text)) - 1 equals the recorded count on EVERY
recorded row.  Eight distinct rows over three orders of magnitude of step
index is a strong constraint: a wrong template is off by a constant, a wrong
BOS/EOS policy is off by one, and a wrong field concatenation is off by a lot.
"""
import json, os, itertools
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

TARGETS = {656: 85, 657: 168, 658: 192, 1679: 118, 1680: 72, 1681: 133,
           7387: 161, 655: 127}

p = hf_hub_download("openai/gsm8k", "main/train-00000-of-00001.parquet",
                    repo_type="dataset")
tbl = pq.read_table(p).to_pylist()
print("rows:", len(tbl), "| cols:", list(tbl[0]))

tok_id = os.environ.get("TOKID", "NousResearch/Meta-Llama-3.1-8B")
tk = AutoTokenizer.from_pretrained(tok_id)
print("tokenizer:", tok_id, "| vocab:", tk.vocab_size, "| bos:", tk.bos_token_id, "eos:", tk.eos_token_id)

TEMPLATES = {
    "q_nl_a":        lambda r: r["question"] + "\n" + r["answer"],
    "q_nlnl_a":      lambda r: r["question"] + "\n\n" + r["answer"],
    "q_sp_a":        lambda r: r["question"] + " " + r["answer"],
    "q_only":        lambda r: r["question"],
    "a_only":        lambda r: r["answer"],
    "qa_labeled":    lambda r: f"Question: {r['question']}\nAnswer: {r['answer']}",
    "qa_labeled_nn": lambda r: f"Question: {r['question']}\n\nAnswer: {r['answer']}",
    "qa_hash":       lambda r: f"{r['question']}\n#### {r['answer']}",
}
# add/remove specials on top of each template
SPECIALS = {
    "plain":   lambda ids: ids,                       # no extra
    "bos":     lambda ids: [tk.bos_token_id] + ids,
    "eos":     lambda ids: ids + [tk.eos_token_id],
    "bos_eos": lambda ids: [tk.bos_token_id] + ids + [tk.eos_token_id],
}

hits = []
for tname, tfn in TEMPLATES.items():
    base = {}
    ok_base = True
    for row, want in TARGETS.items():
        try:
            base[row] = tk(tfn(tbl[row]), add_special_tokens=False)["input_ids"]
        except Exception as e:
            ok_base = False; break
    if not ok_base:
        continue
    for sname, sfn in SPECIALS.items():
        got = {row: len(sfn(ids)) - 1 for row, ids in base.items()}
        diffs = {row: got[row] - TARGETS[row] for row in TARGETS}
        if all(d == 0 for d in diffs.values()):
            hits.append((tname, sname))
            print(f"MATCH  {tname:14s} + {sname:8s}")
        else:
            uniq = set(diffs.values())
            flag = "  (constant offset)" if len(uniq) == 1 else ""
            print(f"  no   {tname:14s} + {sname:8s} diffs={sorted(uniq)[:6]}{flag}")
print()
print("HITS:", hits)
if hits:
    t, s = hits[0]
    ids = SPECIALS[s](tk(TEMPLATES[t](tbl[99]), add_special_tokens=False)["input_ids"])
    print(f"\nrow 99 under {t}+{s}: len(ids)={len(ids)}  supervised={len(ids)-1}")
    print("text:", repr(TEMPLATES[t](tbl[99]))[:400])
