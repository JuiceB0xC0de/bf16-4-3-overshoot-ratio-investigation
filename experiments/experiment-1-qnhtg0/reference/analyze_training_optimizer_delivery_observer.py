"""Offline analyzer for Camera 3 optimizer-delivery bundles (steps 20-25).

Validates a finalized delivery bundle (six NPZs + manifest.json +
storage_report.json + SHA256SUMS + COMPLETED.json) and emits per-optimizer-
parameter delivery summaries plus the control-row detail table.

Semantics guarded here:
  * Three stages are never conflated: pre_clip_parameter_gradient,
    post_clip_parameter_gradient, realized_parameter_delta.
  * The realized delta is a float32 subtraction of native parameters. Deltas
    below the stored dtype's step resolution are exactly zero and are reported
    as fraction_zero -- never imputed, never denoised, never averaged through.
  * The clip coefficient has an EXPECTED view, min(1, max_norm/(returned+1e-6)),
    and an EMPIRICAL view, the global post/pre gradient-norm ratio, plus the
    observer's own pre-clip norm as a third independent view. Material
    disagreement is rejected. Coordinate-wise ratios are never the estimator.
  * The AdamW decomposition is an fp32 SEMANTIC reconstruction of the
    optimizer's intended terms, not a replay of its native instruction path.
    The gap between the idealized total and the realized delta is a
    REALIZATION RESIDUAL containing native optimizer arithmetic and backend
    rounding as well as parameter rounding. It is not called quantization
    error, and no scalar is reported as an "effective learning rate".
  * delivery_control_detail.csv joins to Camera 2 by step/layer/control-family
    identity ONLY. Camera 2 is token x output-feature; Camera 3's weight rows
    are output-feature x input-feature. There is no coordinate-wise join.
  * dt_bias, D and norm.weight full tensors are stored in the bundle and
    summarized per parameter, but are deliberately kept OUT of the Camera-2
    join table: they are adjacent context, not control-region matches.

Stored arrays are never modified. No causal, canary, significance, or
"the optimizer is healthy" claims are made anywhere.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np

OPTIMIZER_DELIVERY_SCHEMA_VERSION = 1
CAMERA_NUMBER = 3
CAMERA_FAMILY = "optimizer_delivery"
CAPTURE_STEPS = tuple(range(20, 26))
PHASE = "optimizer_delivery"

CONTROL_REGIONS = ("dd_dt", "dd_A", "trap", "angles")

DETAIL_FIELDS = (
    "param_pre",
    "grad_pre_clip",
    "grad_post_clip",
    "exp_avg_pre",
    "exp_avg_sq_pre",
    "exp_avg_post",
    "exp_avg_sq_post",
    "delta",
)

PARAM_SCALAR_FIELDS = (
    "pre_clip_grad_l2",
    "pre_clip_grad_max_abs",
    "param_pre_l2",
    "param_pre_max_abs",
    "exp_avg_pre_l2",
    "exp_avg_sq_pre_l2",
    "post_clip_grad_l2",
    "post_clip_grad_max_abs",
    "param_post_l2",
    "param_post_max_abs",
    "exp_avg_post_l2",
    "exp_avg_sq_post_l2",
    "delta_l2",
    "delta_max_abs",
    "delta_fraction_zero",
    "ideal_decay_l2",
    "ideal_adaptive_l2",
    "ideal_total_l2",
    "realization_residual_l2",
    "realization_residual_max_abs",
    "delivery_inner_product",
)
PARAM_GROUP_FLOAT_FIELDS = (
    "group_lr",
    "group_weight_decay",
    "group_beta1",
    "group_beta2",
    "group_eps",
    "bias_correction1",
    "bias_correction2",
)
PARAM_META_INT_FIELDS = (
    "numel",
    "group_index",
    "group_offset",
    "grad_present",
    "state_present_pre",
    "state_present_post",
    "state_step_pre",
    "state_step_post",
    "detail_scope_code",
    "no_weight_decay_hint",
    "amsgrad",
    "maximize",
)
CLIP_GLOBAL_FLOAT_FIELDS = (
    "clip_max_norm",
    "observer_pre_clip_total_norm",
    "returned_total_norm",
    "post_clip_total_norm",
    "clip_coefficient_expected",
    "clip_coefficient_empirical",
    "observer_pre_clip_total_norm_optimizer_scope",
    "post_clip_total_norm_optimizer_scope",
)
CLIP_GLOBAL_INT_FIELDS = (
    "clip_fired",
    "clip_scope_matches_optimizer_scope",
)

GLOBAL_INT_SCALARS = (
    "supervised_token_count",
    "dataset_row",
    "optimizer_step",
    "parameter_count",
)

SUMMARY_FIELDS = (
    "n", "mean", "median", "p05", "p95", "min", "max", "max_abs",
    "l1", "l2", "rms", "fraction_zero",
)

# Quantities summarized per control region in the join table. The last four
# are DERIVED here from the stored sufficient basis, never stored upstream.
DETAIL_QUANTITIES = (
    "delta",
    "grad_pre_clip",
    "grad_post_clip",
    "ideal_decay",
    "ideal_adaptive",
    "ideal_total",
    "realization_residual",
)

CLIP_EPS = 1e-6

# Same derivation as the observer: torch reduces the clipped norm in the
# gradient's own dtype, so the cross-check tolerance follows the COARSEST
# parameter source dtype in the roster rather than being tuned to pass.
#   bfloat16: eps = 2^-8  ~= 3.9e-3   float16: eps = 2^-11 ~= 4.9e-4
#   float32 : eps = 2^-24 ~= 6.0e-8   float64: exact for our purposes
_DTYPE_NORM_RTOL = {
    "torch.bfloat16": 5e-2,
    "torch.float16": 1e-2,
    "torch.float32": 1e-4,
    "torch.float64": 1e-9,
}
_DEFAULT_NORM_RTOL = 5e-2
NORM_CHECK_ATOL = 1e-6
# The expected coefficient is recomputed in float64 from stored float64
# inputs, so it must agree to floating-point round-off, not dtype tolerance.
EXACT_RULE_RTOL = 1e-10


class ArtifactValidationError(RuntimeError):
    """A delivery bundle failed validation. Fail loudly, never impute."""


# ---------------------------------------------------------------------------
# hashing / loading
# ---------------------------------------------------------------------------


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_sha256sums(artifact_dir: Path) -> dict:
    sums_path = artifact_dir / "SHA256SUMS"
    if not sums_path.is_file():
        raise ArtifactValidationError("SHA256SUMS missing from bundle")
    listed = {}
    for line in sums_path.read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split(None, 1)
        name = name.strip()
        if name == "SHA256SUMS":
            raise ArtifactValidationError("SHA256SUMS must not list itself")
        listed[name] = digest
    for name, digest in listed.items():
        target = artifact_dir / name
        if not target.is_file():
            raise ArtifactValidationError(f"SHA256SUMS lists missing file {name}")
        actual = sha256_file(target)
        if actual != digest:
            raise ArtifactValidationError(
                f"SHA-256 mismatch for {name}: listed {digest}, actual {actual}"
            )
    return listed


def _load_npz_checked(path: Path) -> dict:
    """Load one NPZ with allow_pickle=False, rejecting object arrays and
    non-finite float values."""
    arrays = {}
    with np.load(path, allow_pickle=False) as z:
        for key in z.files:
            try:
                arr = z[key]
            except ValueError as exc:
                raise ArtifactValidationError(
                    f"{path.name}:{key} requires pickle to load; refused"
                ) from exc
            if arr.dtype == object or arr.dtype.hasobject:
                raise ArtifactValidationError(
                    f"{path.name}:{key} is an object array; refused"
                )
            if np.issubdtype(arr.dtype, np.floating):
                if not np.all(np.isfinite(arr)):
                    raise ArtifactValidationError(
                        f"{path.name}:{key} contains non-finite values"
                    )
            arrays[key] = arr
    return arrays


def norm_rtol_for_roster(roster: list) -> float:
    """Coarsest source dtype in the roster decides the clip-check tolerance."""
    dtypes = {str(r.get("source_dtype")) for r in roster}
    if not dtypes:
        return _DEFAULT_NORM_RTOL
    return max(_DTYPE_NORM_RTOL.get(d, _DEFAULT_NORM_RTOL) for d in dtypes)


def _close(a: float, b: float, rtol: float, atol: float = NORM_CHECK_ATOL) -> bool:
    return abs(a - b) <= atol + rtol * max(abs(a), abs(b))


# ---------------------------------------------------------------------------
# expected NPZ contents, built from the manifest roster
# ---------------------------------------------------------------------------


def expected_array_spec(roster: list, detail_roster: list) -> dict:
    """key -> (shape or None for 'any 1-D of length P', dtype string)."""
    P = len(roster)
    spec: dict = {}
    for entry in detail_roster:
        base = f"layer_{int(entry['layer']):02d}/{entry['storage_name']}"
        for field in DETAIL_FIELDS:
            spec[f"{base}/{field}"] = (tuple(entry["shape"]), "float32")
    for name in PARAM_SCALAR_FIELDS + PARAM_GROUP_FLOAT_FIELDS:
        spec[f"param/{name}"] = ((P,), "float64")
    for name in PARAM_META_INT_FIELDS:
        spec[f"param_meta/{name}"] = ((P,), "int64")
    for name in CLIP_GLOBAL_FLOAT_FIELDS:
        spec[f"clip/{name}"] = ((), "float64")
    for name in CLIP_GLOBAL_INT_FIELDS:
        spec[f"clip/{name}"] = ((), "int64")
    for name in GLOBAL_INT_SCALARS:
        spec[name] = ((), "int64")
    spec["training_loss"] = ((), "float32")
    spec["token_ids"] = (None, "int64")
    spec["parameter_names"] = ((P,), None)
    return spec


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


class ValidatedBundle:
    def __init__(self, artifact_dir: Path, manifest: dict, completed: dict,
                 steps: dict, geometry: dict, roster: list, detail_roster: list):
        self.artifact_dir = artifact_dir
        self.manifest = manifest
        self.completed = completed
        self.steps = steps          # step -> {"arrays": ..., "seq_len": int}
        self.geometry = geometry
        self.roster = roster
        self.detail_roster = detail_roster
        self.roster_index = {
            r["parameter_id"]: int(r["index"]) for r in roster
        }
        self.norm_rtol = norm_rtol_for_roster(roster)


def _validate_completed(artifact_dir: Path) -> dict:
    completed_path = artifact_dir / "COMPLETED.json"
    if not completed_path.is_file():
        raise ArtifactValidationError(
            "COMPLETED.json missing: bundle is not a finalized capture"
        )
    completed = json.loads(completed_path.read_text())
    if completed.get("success") is not True:
        raise ArtifactValidationError(
            "COMPLETED.json does not declare success=true"
        )
    for key, want in (
        ("optimizer_delivery_schema_version", OPTIMIZER_DELIVERY_SCHEMA_VERSION),
        ("camera_number", CAMERA_NUMBER),
        ("camera_family", CAMERA_FAMILY),
    ):
        if completed.get(key) != want:
            raise ArtifactValidationError(
                f"COMPLETED.json {key}={completed.get(key)!r}, expected {want!r}"
            )
    return completed


def _validate_manifest_semantics(manifest: dict) -> None:
    for key, want in (
        ("optimizer_delivery_schema_version", OPTIMIZER_DELIVERY_SCHEMA_VERSION),
        ("camera_number", CAMERA_NUMBER),
        ("camera_family", CAMERA_FAMILY),
        ("observation_stage", "optimizer_delivery"),
        ("gradient_stage_split", [
            "pre_clip_parameter_gradient",
            "post_clip_parameter_gradient",
            "realized_parameter_delta",
        ]),
        ("delta_precision", "float32_subtraction_of_native_params"),
        ("clip_coefficient_semantics", "expected_and_empirical"),
        ("adamw_decomposition_semantics", "fp32_semantic_reconstruction"),
        ("effective_learning_rate_reported", False),
        ("hooks_registered", 0),
        ("observer_calls_clip_or_step", False),
        ("a_log_parameter_present", False),
    ):
        if manifest.get(key) != want:
            raise ArtifactValidationError(
                f"manifest {key}={manifest.get(key)!r}, expected {want!r}"
            )


def _validate_clip_triad(step: int, arrays: dict, rtol: float) -> dict:
    """The camera's signature integrity check: three independent views of one
    global, uniform operation must agree."""
    max_norm = float(arrays["clip/clip_max_norm"])
    pre = float(arrays["clip/observer_pre_clip_total_norm"])
    returned = float(arrays["clip/returned_total_norm"])
    post = float(arrays["clip/post_clip_total_norm"])
    expected = float(arrays["clip/clip_coefficient_expected"])
    empirical = float(arrays["clip/clip_coefficient_empirical"])
    fired = int(arrays["clip/clip_fired"])

    if pre <= 0.0:
        raise ArtifactValidationError(
            f"step {step}: pre-clip global gradient norm is {pre}; the "
            "empirical clip coefficient would be undefined"
        )

    # (1) The exact clipping rule, recomputed in float64 from stored values.
    rule = min(1.0, max_norm / (returned + CLIP_EPS))
    if not _close(expected, rule, EXACT_RULE_RTOL, atol=1e-12):
        raise ArtifactValidationError(
            f"step {step}: stored expected clip coefficient {expected!r} != "
            f"min(1, {max_norm}/(returned+{CLIP_EPS})) = {rule!r}"
        )

    # (2) Observer's own pre-clip norm vs the norm the real clipper returned.
    if not _close(pre, returned, rtol):
        raise ArtifactValidationError(
            f"step {step}: observer pre-clip total norm {pre!r} disagrees with "
            f"the norm returned by clip_grad_norm_ {returned!r} beyond "
            f"rtol={rtol}"
        )

    # (3) Empirical global post/pre ratio vs the expected coefficient.
    if not _close(empirical, post / pre, EXACT_RULE_RTOL, atol=1e-12):
        raise ArtifactValidationError(
            f"step {step}: stored empirical coefficient {empirical!r} != "
            f"post/pre global norm ratio {post / pre!r}"
        )
    if not _close(empirical, expected, rtol):
        raise ArtifactValidationError(
            f"step {step}: empirical clip coefficient {empirical!r} disagrees "
            f"with expected coefficient {expected!r} beyond rtol={rtol}"
        )
    if fired != int(expected < 1.0):
        raise ArtifactValidationError(
            f"step {step}: clip_fired={fired} contradicts expected "
            f"coefficient {expected!r}"
        )

    # (4) Internal consistency: the stored optimizer-scope globals must be the
    #     norm-of-norms of the stored per-parameter gradient norms.
    for scope_key, per_param in (
        ("observer_pre_clip_total_norm_optimizer_scope", "pre_clip_grad_l2"),
        ("post_clip_total_norm_optimizer_scope", "post_clip_grad_l2"),
    ):
        stored = float(arrays[f"clip/{scope_key}"])
        recomputed = float(np.sqrt(np.sum(np.square(
            np.asarray(arrays[f"param/{per_param}"], dtype=np.float64)
        ))))
        if not _close(stored, recomputed, 1e-9, atol=1e-12):
            raise ArtifactValidationError(
                f"step {step}: {scope_key} {stored!r} != norm-of-norms of "
                f"param/{per_param} {recomputed!r}"
            )

    # (5) When the clipped set and the optimizer set are the same parameters,
    #     the two recorded scopes must be the same number. Without this a
    #     tampered global norm could pass (4) untouched.
    if int(arrays["clip/clip_scope_matches_optimizer_scope"]):
        for total_key, scope_key in (
            ("observer_pre_clip_total_norm",
             "observer_pre_clip_total_norm_optimizer_scope"),
            ("post_clip_total_norm", "post_clip_total_norm_optimizer_scope"),
        ):
            total = float(arrays[f"clip/{total_key}"])
            scoped = float(arrays[f"clip/{scope_key}"])
            if not _close(total, scoped, 1e-12, atol=1e-12):
                raise ArtifactValidationError(
                    f"step {step}: clip scope is declared identical to the "
                    f"optimizer scope but {total_key}={total!r} != "
                    f"{scope_key}={scoped!r}"
                )

    return {
        "clip_max_norm": max_norm,
        "pre_clip_total_norm": pre,
        "returned_total_norm": returned,
        "post_clip_total_norm": post,
        "clip_coefficient_expected": expected,
        "clip_coefficient_empirical": empirical,
        "clip_fired": fired,
        "clip_scope_matches_optimizer_scope": int(
            arrays["clip/clip_scope_matches_optimizer_scope"]
        ),
    }


def _validate_adamw_identities(step: int, arrays: dict, rtol: float) -> dict:
    """Per-parameter AdamW identities checkable from the stored scalars.

    These are the optimizer's own definitions, not fitted relationships.
    """
    grad_present = arrays["param_meta/grad_present"]
    step_post = arrays["param_meta/state_step_post"].astype(np.float64)
    lr = arrays["param/group_lr"]
    wd = arrays["param/group_weight_decay"]
    beta1 = arrays["param/group_beta1"]
    beta2 = arrays["param/group_beta2"]
    bc1 = arrays["param/bias_correction1"]
    bc2 = arrays["param/bias_correction2"]
    param_pre_l2 = arrays["param/param_pre_l2"]
    decay_l2 = arrays["param/ideal_decay_l2"]
    adaptive_l2 = arrays["param/ideal_adaptive_l2"]
    total_l2 = arrays["param/ideal_total_l2"]
    delta_l2 = arrays["param/delta_l2"]
    residual_l2 = arrays["param/realization_residual_l2"]
    zero_fraction = arrays["param/delta_fraction_zero"]

    delivered = grad_present.astype(bool) & (step_post > 0)

    # AdamW skips a grad-less parameter entirely, decoupled decay included.
    skipped = ~grad_present.astype(bool)
    for name, values in (
        ("ideal_decay_l2", decay_l2), ("ideal_adaptive_l2", adaptive_l2),
        ("ideal_total_l2", total_l2), ("delta_l2", delta_l2),
    ):
        if np.any(values[skipped] != 0.0):
            raise ArtifactValidationError(
                f"step {step}: {name} is nonzero for a parameter with no "
                "gradient; AdamW skips those entirely, decay included"
            )

    # Bias corrections are exactly 1 - beta**t.
    for name, beta, stored in (("bias_correction1", beta1, bc1),
                               ("bias_correction2", beta2, bc2)):
        want = 1.0 - np.power(beta[delivered], step_post[delivered])
        if delivered.any() and not np.allclose(
            stored[delivered], want, rtol=1e-9, atol=1e-12
        ):
            raise ArtifactValidationError(
                f"step {step}: {name} does not equal 1 - beta**state_step_post"
            )

    # Decoupled decay is exactly -lr*wd*p_pre, so its norm is lr*wd*||p_pre||.
    want_decay = lr[delivered] * wd[delivered] * param_pre_l2[delivered]
    if delivered.any() and not np.allclose(
        decay_l2[delivered], want_decay, rtol=1e-6, atol=1e-12
    ):
        raise ArtifactValidationError(
            f"step {step}: ideal_decay_l2 does not equal "
            "group_lr * group_weight_decay * param_pre_l2"
        )

    # Triangle inequalities between the idealized parts and the realized delta.
    slack = 1e-6
    if np.any(total_l2 > decay_l2 + adaptive_l2 + slack):
        raise ArtifactValidationError(
            f"step {step}: ideal_total_l2 exceeds ideal_decay_l2 + "
            "ideal_adaptive_l2; the decomposition is inconsistent"
        )
    if np.any(residual_l2 > delta_l2 + total_l2 + slack):
        raise ArtifactValidationError(
            f"step {step}: realization_residual_l2 exceeds delta_l2 + "
            "ideal_total_l2; the residual is inconsistent"
        )
    if np.any((zero_fraction < 0.0) | (zero_fraction > 1.0)):
        raise ArtifactValidationError(
            f"step {step}: delta_fraction_zero outside [0, 1]"
        )

    return {
        "num_parameters_delivered": int(np.count_nonzero(delivered)),
        "num_parameters_grad_absent": int(np.count_nonzero(skipped)),
    }


def validate_bundle(artifact_dir) -> ValidatedBundle:
    artifact_dir = Path(artifact_dir)
    if not artifact_dir.is_dir():
        raise ArtifactValidationError(f"artifact dir not found: {artifact_dir}")

    completed = _validate_completed(artifact_dir)
    listed = verify_sha256sums(artifact_dir)

    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ArtifactValidationError("manifest.json missing")
    if "manifest.json" not in listed:
        raise ArtifactValidationError("manifest.json not covered by SHA256SUMS")
    manifest = json.loads(manifest_path.read_text())
    _validate_manifest_semantics(manifest)

    npz_paths = sorted(artifact_dir.glob("*.npz"))
    expected_names = [f"step_{s:03d}_{PHASE}.npz" for s in CAPTURE_STEPS]
    if [p.name for p in npz_paths] != expected_names:
        raise ArtifactValidationError(
            f"NPZ files {[p.name for p in npz_paths]} != expected {expected_names}"
        )
    for name in expected_names:
        if name not in listed:
            raise ArtifactValidationError(f"{name} not covered by SHA256SUMS")
    if "storage_report.json" not in listed:
        raise ArtifactValidationError(
            "storage_report.json not covered by SHA256SUMS"
        )

    geometry = manifest.get("observer_geometry_discovered")
    if not geometry:
        raise ArtifactValidationError(
            "manifest missing observer_geometry_discovered"
        )
    roster = manifest.get("optimizer_parameter_roster")
    if not roster:
        raise ArtifactValidationError(
            "manifest missing optimizer_parameter_roster"
        )
    detail_roster = manifest.get("detail_roster")
    if detail_roster is None:
        raise ArtifactValidationError("manifest missing detail_roster")

    P = len(roster)
    if [int(r["index"]) for r in roster] != list(range(P)):
        raise ArtifactValidationError(
            "optimizer_parameter_roster indices are not a dense 0..P-1 range"
        )
    ids = [r["parameter_id"] for r in roster]
    if len(set(ids)) != P:
        raise ArtifactValidationError("roster parameter ids are not unique")

    rtol = norm_rtol_for_roster(roster)
    spec = expected_array_spec(roster, detail_roster)
    file_records = {r["filename"]: r for r in manifest.get("files", [])}
    if sorted(file_records) != expected_names:
        raise ArtifactValidationError(
            f"manifest file records {sorted(file_records)} != {expected_names}"
        )

    steps = {}
    for step, path in zip(CAPTURE_STEPS, npz_paths):
        record = file_records[path.name]
        arrays = _load_npz_checked(path)

        missing = sorted(set(spec) - set(arrays))
        if missing:
            raise ArtifactValidationError(
                f"{path.name}: missing arrays {missing[:6]}"
            )
        extra = sorted(set(arrays) - set(spec))
        if extra:
            raise ArtifactValidationError(
                f"{path.name}: unexpected arrays {extra[:6]}"
            )
        for key, (want_shape, want_dtype) in spec.items():
            arr = arrays[key]
            if want_dtype is not None and str(arr.dtype) != want_dtype:
                raise ArtifactValidationError(
                    f"{path.name}:{key} dtype {arr.dtype}, expected {want_dtype}"
                )
            if want_shape is not None and tuple(arr.shape) != want_shape:
                raise ArtifactValidationError(
                    f"{path.name}:{key} shape {tuple(arr.shape)} != expected "
                    f"{want_shape}"
                )

        # Roster order carried in the NPZ must match the manifest roster: a
        # direct guard against roster/array misalignment.
        if list(arrays["parameter_names"]) != ids:
            raise ArtifactValidationError(
                f"{path.name}: parameter_names disagree with the manifest "
                "optimizer_parameter_roster ordering"
            )
        if int(arrays["parameter_count"]) != P:
            raise ArtifactValidationError(
                f"{path.name}: parameter_count {int(arrays['parameter_count'])} "
                f"!= roster size {P}"
            )

        token_ids = arrays["token_ids"]
        if token_ids.ndim != 1:
            raise ArtifactValidationError(f"{path.name}: token_ids not 1-D")
        seq_len = int(token_ids.shape[0]) - 1
        stc = int(arrays["supervised_token_count"])
        if stc != seq_len:
            raise ArtifactValidationError(
                f"{path.name}: supervised_token_count {stc} != "
                f"len(token_ids)-1 = {seq_len}"
            )
        if int(arrays["optimizer_step"]) != step:
            raise ArtifactValidationError(
                f"{path.name}: optimizer_step {int(arrays['optimizer_step'])} "
                f"!= step {step}"
            )
        if int(arrays["dataset_row"]) != step - 1:
            raise ArtifactValidationError(
                f"{path.name}: dataset_row {int(arrays['dataset_row'])} != "
                f"step-1 = {step - 1}"
            )

        token_hash = hashlib.sha256(
            np.ascontiguousarray(token_ids, dtype="<i8").tobytes()
        ).hexdigest()
        if record.get("token_ids_sha256") != token_hash:
            raise ArtifactValidationError(
                f"{path.name}: token_ids hash mismatch vs manifest"
            )
        if int(record.get("sequence_length", -1)) != seq_len:
            raise ArtifactValidationError(
                f"{path.name}: manifest sequence_length "
                f"{record.get('sequence_length')} != T = {seq_len}"
            )

        # byte accounting: manifest tensor raw_bytes must equal array nbytes,
        # and file raw_bytes their sum.
        rec_by_key = {t["key"]: t for t in record["tensors"]}
        if set(rec_by_key) != set(arrays):
            raise ArtifactValidationError(
                f"{path.name}: manifest tensor keys != NPZ keys"
            )
        total_raw = 0
        for key, arr in arrays.items():
            listed_raw = int(rec_by_key[key]["raw_bytes"])
            if listed_raw != int(arr.nbytes):
                raise ArtifactValidationError(
                    f"{path.name}:{key} raw_bytes {listed_raw} != "
                    f"nbytes {int(arr.nbytes)}"
                )
            total_raw += int(arr.nbytes)
        if total_raw != int(record["raw_bytes"]):
            raise ArtifactValidationError(
                f"{path.name}: file raw_bytes {record['raw_bytes']} != "
                f"sum of tensors {total_raw}"
            )
        archive_bytes = int(path.stat().st_size)
        if archive_bytes != int(record["archive_bytes"]):
            raise ArtifactValidationError(
                f"{path.name}: archive_bytes {record['archive_bytes']} != "
                f"on-disk size {archive_bytes}"
            )

        clip = _validate_clip_triad(step, arrays, rtol)
        adamw = _validate_adamw_identities(step, arrays, rtol)

        steps[step] = {
            "arrays": arrays,
            "seq_len": seq_len,
            "record": record,
            "clip": clip,
            "adamw": adamw,
        }

    return ValidatedBundle(
        artifact_dir, manifest, completed, steps, geometry, roster, detail_roster
    )


# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------


def summarize_values(values: np.ndarray) -> dict:
    """Linear summary of a value slice. All-finite is guaranteed by the loader;
    still asserted so a reducer bug cannot smuggle NaN into a report."""
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size == 0:
        raise ArtifactValidationError("empty value slice in summary")
    if not np.all(np.isfinite(v)):
        raise ArtifactValidationError("non-finite value reached summarizer")
    absv = np.abs(v)
    return {
        "n": int(v.size),
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "p05": float(np.percentile(v, 5)),
        "p95": float(np.percentile(v, 95)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "max_abs": float(np.max(absv)),
        "l1": float(np.sum(absv)),
        "l2": float(np.sqrt(np.sum(np.square(v)))),
        "rms": float(np.sqrt(np.mean(np.square(v)))),
        "fraction_zero": float(np.mean(v == 0)),
    }


def _param_rows(bundle: ValidatedBundle) -> list:
    """One row per (step, optimizer parameter). Tied aliases do not duplicate
    rows: the roster is optimizer-centric."""
    rows = []
    alias_by_index = {
        int(r["index"]): "|".join(r.get("aliases") or []) for r in bundle.roster
    }
    for step in CAPTURE_STEPS:
        arrays = bundle.steps[step]["arrays"]
        for record in bundle.roster:
            i = int(record["index"])
            row = {
                "step": step,
                "parameter_index": i,
                "parameter_id": record["parameter_id"],
                "canonical_name": record["canonical_name"],
                "aliases": alias_by_index[i],
                "shape": "x".join(str(s) for s in record["shape"]),
                "source_dtype": record["source_dtype"],
                "detail_scope": record["detail_scope"],
                "detail_layer": record.get("detail_layer"),
            }
            for name in PARAM_META_INT_FIELDS:
                row[name] = int(arrays[f"param_meta/{name}"][i])
            for name in PARAM_GROUP_FLOAT_FIELDS:
                row[name] = float(arrays[f"param/{name}"][i])
            for name in PARAM_SCALAR_FIELDS:
                row[name] = float(arrays[f"param/{name}"][i])

            # Delivery alignment is DERIVED here and left blank where it is
            # genuinely undefined. Writing 0.0 for an undefined cosine would
            # assert orthogonality, which is imputation.
            realized = row["delta_l2"]
            ideal = row["ideal_total_l2"]
            row["delivery_cosine"] = (
                row["delivery_inner_product"] / (realized * ideal)
                if realized > 0.0 and ideal > 0.0 else None
            )
            row["delivery_l2_ratio"] = (
                realized / ideal if ideal > 0.0 else None
            )
            rows.append(row)
    return rows


def _ideal_components(arrays: dict, base: str, index: int) -> dict:
    """Derive the four delivery components offline from the stored sufficient
    basis, using the ACTUAL group settings recorded for the owning parameter.

    AdamW (decoupled):
        p <- p*(1 - lr*wd)              => decay    = -lr*wd*p_pre
        denom = sqrt(v_post)/sqrt(bc2) + eps
        p <- p - (lr/bc1)*m_post/denom  => adaptive = -(lr/bc1)*m_post/denom
    """
    lr = float(arrays["param/group_lr"][index])
    wd = float(arrays["param/group_weight_decay"][index])
    eps = float(arrays["param/group_eps"][index])
    bc1 = float(arrays["param/bias_correction1"][index])
    bc2 = float(arrays["param/bias_correction2"][index])

    p_pre = np.asarray(arrays[f"{base}/param_pre"], dtype=np.float64)
    m_post = np.asarray(arrays[f"{base}/exp_avg_post"], dtype=np.float64)
    v_post = np.asarray(arrays[f"{base}/exp_avg_sq_post"], dtype=np.float64)
    delta = np.asarray(arrays[f"{base}/delta"], dtype=np.float64)

    decay = -(lr * wd) * p_pre
    denom = np.sqrt(v_post) / np.sqrt(bc2) + eps
    adaptive = -(lr / bc1) * m_post / denom
    total = decay + adaptive
    return {
        "ideal_decay": decay,
        "ideal_adaptive": adaptive,
        "ideal_total": total,
        "realization_residual": delta - total,
    }


def _moment_recursion_deviation(arrays: dict, base: str, index: int) -> dict:
    """AdamW's own moment recursion, checked against the stored basis.

    m_post = b1*m_pre + (1-b1)*g_post_clip ; v_post = b2*v_pre + (1-b2)*g^2.

    The deviation is measured against the SUM OF ABSOLUTE TERMS, not against
    the result. Near a cancellation zero the result can be arbitrarily small
    while both terms are large, and dividing by the result would manufacture a
    huge "deviation" out of ordinary rounding. The sum of absolute terms is the
    conditioning scale of the addition, so this stays a real measure of whether
    the stored basis satisfies AdamW's own recursion.
    """
    b1 = float(arrays["param/group_beta1"][index])
    b2 = float(arrays["param/group_beta2"][index])
    g = np.asarray(arrays[f"{base}/grad_post_clip"], dtype=np.float64)
    m_pre = np.asarray(arrays[f"{base}/exp_avg_pre"], dtype=np.float64)
    v_pre = np.asarray(arrays[f"{base}/exp_avg_sq_pre"], dtype=np.float64)
    m_post = np.asarray(arrays[f"{base}/exp_avg_post"], dtype=np.float64)
    v_post = np.asarray(arrays[f"{base}/exp_avg_sq_post"], dtype=np.float64)

    m_terms = (np.abs(b1 * m_pre), np.abs((1.0 - b1) * g))
    v_terms = (np.abs(b2 * v_pre), np.abs((1.0 - b2) * g * g))
    m_want = b1 * m_pre + (1.0 - b1) * g
    v_want = b2 * v_pre + (1.0 - b2) * g * g

    def rel(actual, want, terms):
        scale = terms[0] + terms[1]
        nz = scale > 0
        if not np.any(nz):
            return 0.0
        return float(np.max(np.abs(actual - want)[nz] / scale[nz]))

    return {
        "exp_avg_recursion_max_rel_dev": rel(m_post, m_want, m_terms),
        "exp_avg_sq_recursion_max_rel_dev": rel(v_post, v_want, v_terms),
    }


def _control_detail_rows(bundle: ValidatedBundle) -> list:
    """One row per (step, layer, control region). This is the Camera-2 join
    table: branch identity only, never a coordinate-wise join."""
    control_entries = [
        e for e in bundle.detail_roster
        if e["kind"] == "row_slice" and e["storage_name"] in CONTROL_REGIONS
    ]
    rows = []
    for step in CAPTURE_STEPS:
        arrays = bundle.steps[step]["arrays"]
        for entry in sorted(
            control_entries,
            key=lambda e: (int(e["layer"]), CONTROL_REGIONS.index(e["storage_name"])),
        ):
            layer = int(entry["layer"])
            region = entry["storage_name"]
            base = f"layer_{layer:02d}/{region}"
            index = bundle.roster_index[entry["parameter_id"]]
            if int(arrays["param_meta/amsgrad"][index]) != 0:
                raise ArtifactValidationError(
                    f"step {step} layer {layer} {region}: amsgrad is enabled "
                    "but the v1 detail basis does not store max_exp_avg_sq; "
                    "the adaptive term cannot be reconstructed"
                )
            derived = _ideal_components(arrays, base, index)
            row = {
                "step": step,
                "layer": layer,
                "layer_module_path": entry.get("layer_module_path"),
                "control_region": region,
                "parameter_id": entry["parameter_id"],
                "row_lo": entry["row_lo"],
                "row_hi": entry["row_hi"],
                "rows": int(entry["shape"][0]),
                "cols": int(entry["shape"][1]),
                "state_step_post": int(
                    arrays["param_meta/state_step_post"][index]
                ),
                "clip_coefficient_expected": float(
                    arrays["clip/clip_coefficient_expected"]
                ),
            }
            deviation = _moment_recursion_deviation(arrays, base, index)
            for name, value in deviation.items():
                if value > bundle.norm_rtol:
                    raise ArtifactValidationError(
                        f"step {step} layer {layer} {region}: {name}={value!r} "
                        f"exceeds rtol={bundle.norm_rtol}; the stored basis "
                        "does not satisfy AdamW's own moment recursion"
                    )
            row.update(deviation)
            for quantity in DETAIL_QUANTITIES:
                values = (
                    derived[quantity] if quantity in derived
                    else arrays[f"{base}/{quantity}"]
                )
                for key, value in summarize_values(values).items():
                    row[f"{quantity}_{key}"] = value
            rows.append(row)
    return rows


def _step_overview(bundle: ValidatedBundle) -> list:
    rows = []
    for step in CAPTURE_STEPS:
        info = bundle.steps[step]
        arrays = info["arrays"]
        delta_l2 = np.asarray(arrays["param/delta_l2"], dtype=np.float64)
        residual_l2 = np.asarray(
            arrays["param/realization_residual_l2"], dtype=np.float64
        )
        numel = np.asarray(arrays["param_meta/numel"], dtype=np.float64)
        zero_fraction = np.asarray(
            arrays["param/delta_fraction_zero"], dtype=np.float64
        )
        total_numel = float(np.sum(numel))
        entry = {
            "step": step,
            "training_loss": float(arrays["training_loss"]),
            "supervised_token_count": int(arrays["supervised_token_count"]),
            "sequence_length": info["seq_len"],
            "dataset_row": int(arrays["dataset_row"]),
            "delta_l2_global": float(np.sqrt(np.sum(np.square(delta_l2)))),
            "realization_residual_l2_global": float(
                np.sqrt(np.sum(np.square(residual_l2)))
            ),
            # Element-weighted: the true fraction of parameter elements that
            # did not move, not an unweighted mean over parameters.
            "delta_fraction_zero_global": (
                float(np.sum(zero_fraction * numel) / total_numel)
                if total_numel else 0.0
            ),
            "total_parameter_elements": int(total_numel),
        }
        entry.update(info["clip"])
        entry.update(info["adamw"])
        rows.append(entry)
    return rows


# ---------------------------------------------------------------------------
# output writers
# ---------------------------------------------------------------------------


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return "%.10g" % v
    return v


def _write_csv(path: Path, rows: list, fieldnames: list) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: _fmt(row.get(k)) for k in fieldnames})


def write_outputs(bundle: ValidatedBundle, out_dir) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    param_rows = _param_rows(bundle)
    detail_rows = _control_detail_rows(bundle)
    step_rows = _step_overview(bundle)

    param_fields = (
        ["step", "parameter_index", "parameter_id", "canonical_name", "aliases",
         "shape", "source_dtype", "detail_scope", "detail_layer"]
        + list(PARAM_META_INT_FIELDS)
        + list(PARAM_GROUP_FLOAT_FIELDS)
        + list(PARAM_SCALAR_FIELDS)
        + ["delivery_cosine", "delivery_l2_ratio"]
    )
    _write_csv(out_dir / "delivery_param_metrics.csv", param_rows, param_fields)

    detail_fields = (
        ["step", "layer", "layer_module_path", "control_region", "parameter_id",
         "row_lo", "row_hi", "rows", "cols", "state_step_post",
         "clip_coefficient_expected", "exp_avg_recursion_max_rel_dev",
         "exp_avg_sq_recursion_max_rel_dev"]
        + [f"{q}_{f}" for q in DETAIL_QUANTITIES for f in SUMMARY_FIELDS]
    )
    _write_csv(
        out_dir / "delivery_control_detail.csv", detail_rows, detail_fields
    )

    storage_report = json.loads(
        (bundle.artifact_dir / "storage_report.json").read_text()
    )
    _write_csv(
        out_dir / "storage_by_quantity.csv",
        storage_report.get("by_quantity", []),
        ["name", "count", "raw_bytes", "npy_bytes", "compressed_bytes"],
    )

    num_layers = int(bundle.geometry["num_layers"])
    P = len(bundle.roster)
    scope_counts: dict = {}
    for record in bundle.roster:
        scope_counts[record["detail_scope"]] = (
            scope_counts.get(record["detail_scope"], 0) + 1
        )

    analysis = {
        "optimizer_delivery_schema_version": OPTIMIZER_DELIVERY_SCHEMA_VERSION,
        "camera_number": CAMERA_NUMBER,
        "camera_family": CAMERA_FAMILY,
        "observation_stage": "optimizer_delivery",
        "gradient_stage_split": list(bundle.manifest["gradient_stage_split"]),
        "delta_precision": bundle.manifest["delta_precision"],
        "clip_coefficient_semantics": (
            "expected = min(1, max_norm/(returned_total_norm + 1e-6)); "
            "empirical = global post/pre gradient-norm ratio; the observer's "
            "own pre-clip norm is a third independent view; coordinate-wise "
            "ratios are never used as the estimator"
        ),
        "adamw_decomposition_semantics": (
            "fp32 semantic reconstruction of AdamW's intended decoupled-decay "
            "and adaptive terms from the stored sufficient basis and the "
            "ACTUAL group settings; not a replay of the native instruction "
            "path"
        ),
        "realization_residual_semantics": (
            "realized_delta - ideal_total_delta; contains native optimizer "
            "arithmetic and backend rounding as well as parameter rounding; "
            "it is NOT pure bf16 quantization error"
        ),
        "effective_learning_rate_reported": False,
        "bf16_floor_semantics": (
            "deltas below the stored parameter dtype's step resolution are "
            "exactly zero and are reported as fraction_zero; never imputed, "
            "never denoised, never averaged through"
        ),
        "delivery_alignment_semantics": (
            "delivery_cosine and delivery_l2_ratio are derived here and left "
            "BLANK where the norms make them undefined; no stand-in value is "
            "written"
        ),
        "camera2_join_semantics": (
            "delivery_control_detail.csv joins to Camera 2 by "
            "step/layer/control-family identity only; Camera 2 is token x "
            "output-feature and Camera 3's weight rows are output-feature x "
            "input-feature, so there is no coordinate-wise join"
        ),
        "adjacent_detail_note": (
            "dt_bias, D and norm.weight full tensors are stored in the bundle "
            "and summarized per parameter, but are deliberately excluded from "
            "the Camera-2 join table: they are adjacent context, not control "
            "regions"
        ),
        "interpretation_limits": (
            "the delta is what happened on this step, not what would have "
            "happened under a different optimizer, clip setting, or moment "
            "state; no causal, canary, significance, or optimizer-health "
            "claims"
        ),
        "clip_check_rtol": bundle.norm_rtol,
        "geometry": bundle.geometry,
        "num_layers": num_layers,
        "num_optimizer_parameters": P,
        "detail_scope_counts": scope_counts,
        "num_control_regions": len(CONTROL_REGIONS),
        "num_detail_tensors": len(bundle.detail_roster),
        "steps": step_rows,
        "num_param_metric_rows": len(param_rows),
        "num_control_detail_rows": len(detail_rows),
        "moment_recursion_max_rel_dev": max(
            (max(r["exp_avg_recursion_max_rel_dev"],
                 r["exp_avg_sq_recursion_max_rel_dev"]) for r in detail_rows),
            default=0.0,
        ),
        "param_row_formula": "steps x optimizer parameters",
        "control_detail_row_formula": "steps x layers x control regions",
        "validated_files": sorted(
            p.name for p in bundle.artifact_dir.glob("*.npz")
        ),
        "manifest_wandb_run_id": bundle.manifest.get("wandb_run_id"),
        "completed_wandb_run_id": bundle.completed.get("wandb_run_id"),
        "total_raw_bytes": bundle.manifest.get("total_raw_bytes"),
        "total_compressed_bytes": bundle.manifest.get("total_compressed_bytes"),
    }
    (out_dir / "optimizer_delivery_analysis.json").write_text(
        json.dumps(analysis, sort_keys=True, indent=2) + "\n"
    )

    md = io.StringIO()
    md.write("# Camera 3 Optimizer Delivery analysis\n\n")
    md.write(
        "Parameter gradients before and after the trainer's single "
        "`clip_grad_norm_`, the AdamW state around its single "
        "`optimizer.step()`, and the realized parameter delta, captured "
        "during optimizer updates 20-25.\n\n"
        "**Scope**: Camera 3 is read-only -- it registers no hooks and calls "
        "neither the clipper nor the optimizer. The realized delta is a "
        "float32 subtraction of native parameters; deltas below the stored "
        "dtype's step resolution are exactly zero and are reported as "
        "`fraction_zero`, never imputed. The idealized decay and adaptive "
        "terms are an fp32 *semantic* reconstruction; the gap to the realized "
        "delta is a **realization residual** that also contains native "
        "optimizer and backend arithmetic, and is not quantization error. No "
        "effective learning rate is reported: AdamW's elementwise "
        "normalization does not have one.\n\n"
        "This camera is one component of the pre-canary observer suite. It "
        "makes no causal, canary, significance, or optimizer-health claims.\n\n"
    )
    md.write("## Per-step clip event and delivery\n\n")
    cols = [
        "step", "training_loss", "pre_clip_total_norm", "returned_total_norm",
        "clip_coefficient_expected", "clip_coefficient_empirical",
        "clip_fired", "delta_l2_global", "delta_fraction_zero_global",
        "realization_residual_l2_global",
    ]
    md.write("| " + " | ".join(cols) + " |\n")
    md.write("|" + "---|" * len(cols) + "\n")
    for row in step_rows:
        md.write("| " + " | ".join(str(_fmt(row[c])) for c in cols) + " |\n")
    md.write(
        f"\nParameter metric rows: {len(param_rows)} "
        f"({len(CAPTURE_STEPS)} steps x {P} optimizer parameters; tied "
        "aliases do not duplicate rows). "
        f"Control detail rows: {len(detail_rows)} "
        f"({len(CAPTURE_STEPS)} steps x {num_layers} layers x "
        f"{len(CONTROL_REGIONS)} control regions).\n\n"
        "`delivery_control_detail.csv` is joinable to Camera 2 by "
        "step/layer/control-family identity only -- Camera 2 is token x "
        "output-feature, Camera 3's weight rows are output-feature x "
        "input-feature.\n"
    )
    (out_dir / "optimizer_delivery_analysis.md").write_text(md.getvalue())

    return analysis


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate and summarize a Camera 3 optimizer-delivery bundle."
    )
    parser.add_argument("--artifact-dir", required=True,
                        help="finalized bundle directory (with COMPLETED.json)")
    parser.add_argument("--out-dir", required=True,
                        help="output directory for analysis files")
    args = parser.parse_args(argv)

    bundle = validate_bundle(args.artifact_dir)
    analysis = write_outputs(bundle, args.out_dir)
    print(
        f"validated {len(bundle.steps)} steps; "
        f"param rows={analysis['num_param_metric_rows']} "
        f"control detail rows={analysis['num_control_detail_rows']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
