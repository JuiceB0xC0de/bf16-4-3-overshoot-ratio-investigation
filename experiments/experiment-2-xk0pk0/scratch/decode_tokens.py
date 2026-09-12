"""Read token_ids straight out of the recorded step-100 camera-3 bundle.

An NPZ is a zip; HfFileSystem gives a seekable handle, so only the token_ids
member is transferred.  This is the definitive answer to what text entered the
forward pass -- no template guessing.
"""
import numpy as np, json
from huggingface_hub import HfFileSystem, hf_hub_download

DATASET = "juiceb0xc0de/mmb3-slt-6xcam-results"
fs = HfFileSystem()
base = f"datasets/{DATASET}/observed-runs/20260810T091311Z/e1_s000100/camera-3"
names = [p["name"] for p in fs.ls(base, detail=True)]
print("bundle files:", [n.rsplit("/",1)[1] for n in names])
npz = [n for n in names if n.endswith("_optimizer_delivery.npz")][0]
with fs.open(npz, "rb") as fh:
    z = np.load(fh)
    keys = [k for k in z.files if "token" in k or k in ("dataset_row","optimizer_step","supervised_token_count","training_loss")]
    print("keys:", keys)
    ids = z["token_ids"]
    row = int(z["dataset_row"]); stc = int(z["supervised_token_count"])
    loss = float(z["training_loss"])
print(f"row={row} len(ids)={len(ids)} supervised={stc} loss={loss}")
print("first 24:", ids[:24].tolist())
print("last 12:", ids[-12:].tolist())
np.save("step100_token_ids.npy", np.asarray(ids, dtype=np.int64))

from transformers import AutoTokenizer
tk = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3.1-8B")
print("\n--- decoded (with specials) ---")
print(repr(tk.decode(ids, skip_special_tokens=False)))
print("\n--- per-token head/tail ---")
for i in list(range(8)) + list(range(len(ids)-8, len(ids))):
    print(f"  {i:4d} {int(ids[i]):7d} {tk.convert_ids_to_tokens([int(ids[i])])[0]!r}")
