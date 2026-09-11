"""Arm A driver: fetch the Camera-3 bundles, replay each through five writers.

    python run_arm_a.py --out results/arm_a.json

A bundle whose files do not match its own SHA256SUMS is dropped and reported,
never repaired.  A bundle whose manifest identity disagrees with the run it was
requested from is dropped the same way: the directory label is a label, the
manifest is the authority.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bundles import (  # noqa: E402
    DATASET,
    fetch_bundle,
    npz_path,
    planned_bundles,
    read_identity,
)
from replay import BundleCorrupt, replay_capture, sha256_file, verify_dir  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("bundle-cache"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 = every planned step")
    parser.add_argument("--only-run", default="", help="restrict to one run label")
    parser.add_argument("--keep-bundles", action="store_true")
    args = parser.parse_args()

    args.cache.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    planned = planned_bundles()
    if args.only_run:
        planned = [p for p in planned if p["run_label"] == args.only_run]
    if args.limit:
        planned = planned[: args.limit]

    records, dropped = [], []
    started = time.time()
    for position, plan in enumerate(planned, start=1):
        label = f"{plan['run_label']}/{plan['step_dir']}"
        try:
            directory = fetch_bundle(plan["remote"], args.cache)
            verify_dir(directory)
            identity = read_identity(directory)
            if identity["run_id"] != plan["run_id"]:
                raise BundleCorrupt(
                    f"manifest run_id {identity['run_id']!r} is not the requested "
                    f"{plan['run_id']!r}"
                )
            if not identity["finite_assertions_passed"]:
                raise BundleCorrupt("capture did not pass its own finite assertions")
            archive = npz_path(directory)
            record = replay_capture(archive, seed=args.seed, bootstrap=args.bootstrap)
        except Exception as error:  # noqa: BLE001 - the drop reason is the product
            dropped.append({"bundle": label, "reason": f"{type(error).__name__}: {error}"})
            print(f"  DROP {label}: {type(error).__name__}: {error}", flush=True)
            continue

        record.update(
            {
                "run_label": plan["run_label"],
                "run_id": plan["run_id"],
                "model": plan["model"],
                "grad_clip_norm": plan["grad_clip_norm"],
                "step_dir": plan["step_dir"],
                "remote": plan["remote"],
                "identity": identity,
                "npz_sha256": sha256_file(archive),
            }
        )
        records.append(record)
        fidelity = record["fidelity"]
        observed = record["observed"]["detail_all"]["overshoot"]
        print(
            f"  [{position}/{len(planned)}] {label}: step {record['optimizer_step']:>6} "
            f"coords {record['n_detail_coords']:>8}  observed overshoot {observed:.5f}  "
            f"bf16 replay {record['writers']['bf16_nearest_even']['detail_all']['overshoot']:.5f}  "
            f"fp32 {record['writers']['fp32']['detail_all']['overshoot']:.5f}  "
            f"match {fidelity['match_rate_all']:.5f}",
            flush=True,
        )
        if not args.keep_bundles:
            for path in directory.iterdir():
                path.unlink()
            directory.rmdir()

    payload = {
        "schema": "arm-a-writer-replay/1",
        "dataset": DATASET,
        "seed": args.seed,
        "n_planned": len(planned),
        "n_records": len(records),
        "dropped": dropped,
        "elapsed_seconds": time.time() - started,
        "numpy_version": __import__("numpy").__version__,
        "python_version": sys.version,
        "steps": records,
    }
    args.out.write_text(json.dumps(payload, indent=1))
    print(
        f"\nwrote {args.out}  ({len(records)}/{len(planned)} bundles, "
        f"{len(dropped)} dropped, {payload['elapsed_seconds']:.0f}s)"
    )


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    main()
