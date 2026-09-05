"""Which step bundles Arm A reads, and how they are fetched and verified.

The steps are fixed here rather than chosen after looking at results.  Twelve
steps from each 187M run and four from the 443M run, spread evenly over the
captured range and always including step 100, which is the step the reference
numbers in the plan were measured on.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

DATASET = "juiceb0xc0de/mmb3-slt-6xcam-results"
CAMERA = "camera-3"
BUNDLE_FILES = ("SHA256SUMS", "manifest.json", "storage_report.json")

_CLIP1_STEPS = [
    "e1_s000100", "e1_s000800", "e1_s001500", "e1_s002400", "e1_s003300",
    "e1_s004400", "e1_s005500", "e1_s006600", "e1_s007400",
    "e2_s001500", "e2_s004000", "e2_s007400",
]
_CLIP8_STEPS = list(_CLIP1_STEPS)
_M443_STEPS = ["step_000100", "step_001300", "step_002600", "step_004000"]

RUNS = {
    "187M_clip1": {
        "prefix": "observed-runs/20260810T091311Z",
        "run_id": "20260810T091311Z",
        "model": "mamba3-siso-187m",
        "grad_clip_norm": 1.0,
        "steps": _CLIP1_STEPS,
    },
    "187M_clip8": {
        # Directory label and run_id differ in this dataset; the run_id in the
        # manifests is the authority and is checked at load time.
        "prefix": "observed-runs/second-pass_clip8_148-observations",
        "run_id": "20260811T080804Z",
        "model": "mamba3-siso-187m",
        "grad_clip_norm": 8.0,
        "steps": _CLIP8_STEPS,
    },
    "443M": {
        "prefix": "443m_run-1",
        "run_id": "20260814T133327Z",
        "model": "mamba3-siso-443m",
        "grad_clip_norm": 8.0,
        "steps": _M443_STEPS,
    },
}


def planned_bundles() -> list[dict]:
    out = []
    for label, run in RUNS.items():
        for step_dir in run["steps"]:
            out.append(
                {
                    "run_label": label,
                    "run_id": run["run_id"],
                    "model": run["model"],
                    "grad_clip_norm": run["grad_clip_norm"],
                    "step_dir": step_dir,
                    "remote": f"{run['prefix']}/{step_dir}/{CAMERA}",
                }
            )
    return out


def fetch_bundle(remote: str, destination: Path) -> Path:
    """Download one camera directory intact, so its own SHA256SUMS can gate it.

    Every file the checksum manifest names is fetched, including the ones this
    analysis does not read: a partial directory would make ``verify_dir``
    report a missing file and the bundle would be dropped for the wrong reason.
    """
    from huggingface_hub import hf_hub_download

    directory = destination / remote.replace("/", "__")
    directory.mkdir(parents=True, exist_ok=True)

    sums = Path(
        hf_hub_download(DATASET, f"{remote}/SHA256SUMS", repo_type="dataset")
    )
    names = [line.split("  ", 1)[1] for line in sums.read_text().strip().splitlines()]
    shutil.copyfile(sums, directory / "SHA256SUMS")
    for name in names:
        local = Path(
            hf_hub_download(DATASET, f"{remote}/{name}", repo_type="dataset")
        )
        target = directory / name
        if not target.exists() or target.stat().st_size != local.stat().st_size:
            shutil.copyfile(local, target)
    return directory


def read_identity(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    return {
        key: manifest.get(key)
        for key in (
            "run_id", "global_step", "local_step", "epoch", "row_index",
            "model_id", "model_revision", "grad_clip_norm", "lr", "image_ref",
            "finite_assertions_passed", "delta_precision",
            "adamw_decomposition_semantics", "optimizer_delivery_schema_version",
        )
    }


def npz_path(directory: Path) -> Path:
    hits = sorted(directory.glob("*_optimizer_delivery.npz"))
    if not hits:
        raise FileNotFoundError(f"no optimizer delivery archive in {directory}")
    return hits[0]
