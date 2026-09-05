"""Maximally diagnostic environment report: every stage runs, none aborts.

Run first in a batch job.  Every stage is wrapped so a failure is *recorded*
and execution continues, because the expensive thing about a batch lane is the
round trip: learning about one missing piece per submission is the failure mode
this file exists to prevent.

Always exits 0.  The caller decides what to do with ``all_ok`` in the report;
the point here is that one job produces the complete picture.

    python3 src/preflight_report.py [--out preflight_report.json]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

STAGES = []
# Stages hand objects to later stages through this, not through function
# attributes: the decorator runs each stage at decoration time, so a stage's
# own name is not yet bound inside its body.
CTX = {}


def stage(name: str, critical: bool = True):
    """Run a stage, record whatever happened, and keep going."""
    def wrap(fn):
        started = time.time()
        record = {"stage": name, "critical": critical}
        try:
            record["detail"] = fn()
            record["ok"] = True
        except BaseException as exc:                       # noqa: BLE001
            record["ok"] = False
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["traceback"] = traceback.format_exc()[-4000:]
        record["seconds"] = round(time.time() - started, 3)
        STAGES.append(record)
        flag = "ok  " if record["ok"] else "FAIL"
        print(f"[{flag}] {name}  ({record['seconds']}s)", flush=True)
        if record["ok"]:
            print(f"        {json.dumps(record['detail'])[:1400]}", flush=True)
        else:
            print(f"        {record['error']}", flush=True)
            for line in record["traceback"].strip().splitlines()[-6:]:
                print(f"        | {line}", flush=True)
        return fn
    return wrap


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="preflight_report.json")
    args = ap.parse_args()

    print("=" * 72, flush=True)
    print("ARM B BATCH-LANE DIAGNOSTIC -- every stage runs, none aborts", flush=True)
    print("=" * 72, flush=True)

    @stage("interpreter")
    def _interpreter():
        return {
            "python_version": sys.version.split()[0],
            "executable": sys.executable,
            "platform": platform.platform(),
            "cwd": os.getcwd(),
            "argv0_dir": str(Path(__file__).resolve().parent),
        }

    @stage("environment")
    def _environment():
        import shutil

        want = ["SILICO_EXPERIMENT_ARTIFACTS_DIR", "SILICO_EXPERIMENT_RELATIVE_DIR",
                "SILICO_EXPERIMENT_ID", "HF_HOME", "PATH", "PYTHONPATH"]
        # Credential PRESENCE only -- never the value.
        creds = {k: bool(os.environ.get(k)) for k in ("HF_TOKEN", "WANDB_API_KEY")}
        art = os.environ.get("SILICO_EXPERIMENT_ARTIFACTS_DIR", "")
        usage = {}
        for path in [p for p in ("/", "/tmp", os.getcwd(), art) if p and Path(p).exists()]:
            total, used, free = shutil.disk_usage(path)
            usage[path] = {"total_gb": round(total / 1e9, 1), "free_gb": round(free / 1e9, 1)}
        return {
            "vars": {k: os.environ.get(k) for k in want},
            "credentials_present": creds,
            "disk": usage,
        }

    @stage("artifacts_dir_writable")
    def _artifacts():
        art = Path(os.environ.get("SILICO_EXPERIMENT_ARTIFACTS_DIR", "./artifacts"))
        art.mkdir(parents=True, exist_ok=True)
        probe = art / ".preflight_write_probe"
        probe.write_text("ok")
        content = probe.read_text()
        probe.unlink()
        return {"dir": str(art), "roundtrip_ok": content == "ok"}

    @stage("torch")
    def _torch():
        import torch

        return {
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "device_name": (torch.cuda.get_device_name(0)
                            if torch.cuda.is_available() else None),
            "capability": (list(torch.cuda.get_device_capability(0))
                           if torch.cuda.is_available() else None),
            "total_memory_gb": (round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
                                if torch.cuda.is_available() else None),
        }

    @stage("torch_gpu_op")
    def _torch_gpu():
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA device visible to torch")
        a = torch.randn(512, 512, device="cuda")
        fp32 = float((a @ a.T).sum())
        b = a.to(torch.bfloat16)
        bf16 = float((b @ b.T).float().sum())
        # bf16 round-trip is the exact operation the measurement depends on.
        x = torch.tensor([1.0009765625, 3.14159, -0.02], device="cuda")
        rt = x.to(torch.bfloat16).to(torch.float32)
        torch.cuda.synchronize()
        return {"fp32_matmul_finite": bool(fp32 == fp32),
                "bf16_matmul_finite": bool(bf16 == bf16),
                "bf16_roundtrip": rt.tolist(),
                "peak_mem_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1)}

    @stage("mamba_ssm")
    def _mamba():
        import importlib

        import mamba_ssm

        kernels = {}
        for mod in ("mamba_ssm.ops.selective_scan_interface",
                    "mamba_ssm.ops.triton.ssd_combined",
                    "mamba_ssm.ops.triton.layernorm_gated",
                    "mamba_ssm.modules.mamba3",
                    "mamba_ssm.models.mixer_seq_simple",
                    "selective_scan_cuda", "causal_conv1d_cuda"):
            try:
                importlib.import_module(mod)
                kernels[mod] = "ok"
            except BaseException as exc:                   # noqa: BLE001
                kernels[mod] = f"{type(exc).__name__}: {exc}"[:160]
        import inspect

        from mamba_ssm.models import mixer_seq_simple as m
        return {
            "version": mamba_ssm.__version__,
            "imports": kernels,
            "Mamba3_in_create_block": "Mamba3" in inspect.getsource(m.create_block),
        }

    @stage("other_libraries")
    def _libs():
        import importlib

        out = {}
        for mod in ("transformers", "datasets", "huggingface_hub", "numpy",
                    "pyarrow", "wandb"):
            try:
                out[mod] = getattr(importlib.import_module(mod), "__version__", "?")
            except BaseException as exc:                   # noqa: BLE001
                out[mod] = f"MISSING: {type(exc).__name__}"
        return out

    @stage("data_pipeline")
    def _data():
        from data import StepData

        data = StepData()
        report = data.verify()
        report.update(data.window_stats(300))
        return report

    @stage("model_load_and_roster")
    def _model():
        import conditions as C

        C.set_seed()
        model, raw = C.load_model()
        records = C.build_roster(model)
        CTX["model"] = model
        CTX["records"] = records
        return {"config": raw, "roster": C.roster_summary(records)}

    @stage("one_training_step")
    def _step():
        import torch

        import conditions as C
        from data import StepData
        from observer import DeliveryObserver, grad_stats
        from run_arm_b import build_loss

        model, records = CTX["model"], CTX["records"]
        condition = C.Condition("adamw", records, lr=C.ADAMW_LR)
        data = StepData()
        ids = torch.tensor([data.ids_for_step(1)], device="cuda")
        condition.zero_grad()
        loss, loss_f32, loss_native = build_loss(model, ids, "float32")
        loss.backward()
        pre = grad_stats(records, C.CLIP_MAX_NORM)
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                    max_norm=C.CLIP_MAX_NORM))
        observer = DeliveryObserver(records, condition, torch.device("cuda"))
        observer.snapshot()
        condition.step()
        CTX["observer"] = observer
        return {"loss_f32": float(loss_f32.detach()),
                "loss_native": float(loss_native),
                "grad_norm_preclip": norm,
                "clip_fired": int(norm > C.CLIP_MAX_NORM), **pre}

    @stage("one_capture_end_to_end")
    def _capture():
        capture = CTX["observer"].capture(want_geometry=True, want_per_tensor=False)
        pooled = capture["pooled"]
        geo = capture["geometry"]["muon_covered"]
        return {
            "pooled": {k: {kk: v[kk] for kk in
                           ("ideal", "realized", "delivery", "cosine",
                            "overshoot", "zero_fraction", "n_coords")}
                       for k, v in pooled.items()},
            "muon_covered_geometry": {
                k: geo[k] for k in
                ("x_mean", "zero_step_fraction", "abs_step_ge_2_share_of_moved",
                 "ratio_predicted_from_x", "ratio_from_realized_steps",
                 "decay_x_max", "uniform_x_reference")},
            "step_hist_counts": geo["step_hist"]["counts"],
        }

    @stage("shadow_one_step", critical=False)
    def _shadow():
        import torch

        from shadow import REFERENCE_MULTIPLIER, ShadowOptimizers

        records = CTX["records"]
        shadow = ShadowOptimizers(records)
        pre = {r.index: r.param.detach().clone() for r in records if r.muon_covered}
        grads = {r.index: (r.param.grad.detach().clone()
                           if r.param.grad is not None
                           else torch.zeros_like(r.param))
                 for r in records if r.muon_covered}
        out = shadow.measure(pre, grads, 1.0, want_geometry=False)
        ref = f"{REFERENCE_MULTIPLIER:.6g}"
        return {name: {"overshoot": out[name][ref]["overshoot"],
                       "n_scales": len(out[name])} for name in out}

    ok = all(s["ok"] for s in STAGES)
    critical_ok = all(s["ok"] for s in STAGES if s["critical"])
    report = {
        "schema": "arm-b-preflight/1",
        "all_ok": ok,
        "critical_ok": critical_ok,
        "n_stages": len(STAGES),
        "n_failed": sum(1 for s in STAGES if not s["ok"]),
        "failed_stages": [s["stage"] for s in STAGES if not s["ok"]],
        "stages": STAGES,
    }
    art = Path(os.environ.get("SILICO_EXPERIMENT_ARTIFACTS_DIR", "."))
    art.mkdir(parents=True, exist_ok=True)
    (art / args.out).write_text(json.dumps(report, indent=1))

    print("=" * 72, flush=True)
    print(f"SUMMARY  {len(STAGES) - report['n_failed']}/{len(STAGES)} stages ok"
          f"  | critical_ok={critical_ok}", flush=True)
    if report["failed_stages"]:
        print(f"FAILED:  {report['failed_stages']}", flush=True)
    print(f"report -> {art / args.out}", flush=True)
    print("=" * 72, flush=True)
    sys.exit(0)          # never abort the job: the caller reads the report


if __name__ == "__main__":
    main()
