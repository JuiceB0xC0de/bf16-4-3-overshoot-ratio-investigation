import numpy as np, pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

TARGETS = {656: 85, 657: 168, 658: 192, 1679: 118, 1680: 72, 1681: 133, 7387: 161, 655: 127}
TEMPLATE = "Question:\n{q}\n\nAnswer:\n{a}"

rows = pq.read_table(hf_hub_download("openai/gsm8k", "main/train-00000-of-00001.parquet",
                                     repo_type="dataset")).to_pylist()
want = np.load("step100_token_ids.npy")

for tok_id in ["NousResearch/Meta-Llama-3.1-8B", "meta-llama/Llama-3.1-8B"]:
    try:
        tk = AutoTokenizer.from_pretrained(tok_id)
    except Exception as e:
        print(f"{tok_id}: UNAVAILABLE ({type(e).__name__})"); continue

    def encode(r):
        body = TEMPLATE.format(q=r["question"], a=r["answer"])
        return [tk.bos_token_id] + tk(body, add_special_tokens=False)["input_ids"] + [tk.eos_token_id]

    got = encode(rows[99])
    exact = (len(got) == len(want)) and bool(np.array_equal(np.asarray(got), want))
    counts_ok = all(len(encode(rows[r])) - 1 == n for r, n in TARGETS.items())
    print(f"{tok_id}:")
    print(f"   row99 token_ids EXACT MATCH: {exact}  (len {len(got)} vs {len(want)})")
    print(f"   all 8 recorded n_tokens match: {counts_ok}")
    if not counts_ok:
        for r, n in TARGETS.items():
            print(f"     row {r}: got {len(encode(rows[r]))-1} want {n}")
    lens = [len(encode(rows[i])) for i in range(300)]
    print(f"   first 300 rows: len min={min(lens)} med={int(np.median(lens))} max={max(lens)} "
          f"| n>512: {sum(l>512 for l in lens)}")
