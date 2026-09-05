"""Replay one Camera-3 capture through a different parameter writer.

The capture stores, per detailed coordinate, the pre-update parameter, both
post-update AdamW moments, and the realized delta.  That is a sufficient basis
to reconstruct the *intended* update exactly as the project's own analyzer
defines it, and then to re-apply that same intent through a different storage
format or rounding rule.  Only the writer changes; the intent is byte-identical
across writers.

Two definitions carry the whole experiment and both are taken from the
project's own code rather than restated:

* intent, from ``analyze_training_optimizer_delivery_observer._ideal_components``
      decay    = -(lr * wd) * p_pre
      adaptive = -(lr / bc1) * m_post / (sqrt(v_post) / sqrt(bc2) + eps)
* pooling, from ``transmission_extract._pair``
      delivery = |r| / |i| ,  cosine = <i, r> / (|i| |r|)
      overshoot = delivery / cosine = |r|^2 / <i, r>

The write itself is modelled as float32 arithmetic followed by a store to the
writer's lattice, which is what the observed pipeline did
(``adamw_decomposition_semantics = "fp32_semantic_reconstruction"``).  Each
writer's starting point is projected onto its own lattice first, so a
counterfactual writer is asked "what would this format have done from a state
it could actually hold", not "round a value you cannot represent".  For the
native writer on a native parameter that projection is the identity, so the
fidelity check is unaffected by it.
"""

from __future__ import annotations

import hashlib
import math
import re
import sys
import types
from pathlib import Path

import numpy as np

from writers import PROJECTIONS, ULP, bf16_nearest_even, bf16_ulp, fp32, writer_table

# transmission_extract imports orjson only for its CLI; stub it so the pooling
# and verification semantics can be imported from the project's own source
# rather than restated here.
if "orjson" not in sys.modules:
    try:
        import orjson  # noqa: F401
    except ModuleNotFoundError:
        sys.modules["orjson"] = types.ModuleType("orjson")

_REFERENCE = Path(__file__).resolve().parent.parent / "reference"
sys.path.insert(0, str(_REFERENCE))
from analyze_training_optimizer_delivery_observer import CONTROL_REGIONS  # noqa: E402
from transmission_extract import (  # noqa: E402
    FP32_SUFFIXES,
    BundleCorrupt,
    _pair,
    verify_dir,  # noqa: F401 - re-exported so the driver uses the project's own gate
)

DETAIL_FIELDS_NEEDED = ("param_pre", "exp_avg_post", "exp_avg_sq_post", "delta")

# Detail bases that are whole tensors rather than in_proj row slices, and the
# suffix that identifies the parameter each was sliced from.  Matching by
# pattern rather than by a fixed string keeps the 187M and 443M checkpoints on
# the same code path even if their module paths differ.
FULL_TENSOR_SUFFIX = {
    "dt_bias": r"\.dt_bias$",
    "D": r"\.D$",
    "norm_weight": r"norm\.weight$",
}
ROW_SLICE_SUFFIX = r"in_proj\.weight$"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def owning_parameter(base: str, names: list[str]) -> str:
    """Map a detail storage base to the parameter it was sliced from.

    The answer is the unique parameter in that layer whose name ends the right
    way; an ambiguous or absent match is an error rather than a guess, because
    a detail vector attributed to the wrong parameter would be reconstructed
    with the wrong learning rate and weight decay.  Matching by pattern keeps
    the 187M and 443M checkpoints on one code path.
    """
    layer_token, region = base.split("/")
    layer = int(layer_token.split("_")[1])
    if region in CONTROL_REGIONS:
        suffix = ROW_SLICE_SUFFIX
    elif region in FULL_TENSOR_SUFFIX:
        suffix = FULL_TENSOR_SUFFIX[region]
    else:
        raise KeyError(f"unknown detail region {region!r}")

    pattern = re.compile(rf"(^|\.)layers\.{layer}\..*{suffix}")
    hits = [name for name in names if pattern.search(name)]
    if len(hits) != 1:
        raise BundleCorrupt(
            f"detail base {base} matches {len(hits)} parameters {hits[:4]}"
        )
    return hits[0]


def intended_update(archive, base: str, index: int) -> np.ndarray:
    """The AdamW update the optimizer meant to apply, in float64.

    Identical in form to the project analyzer's ``_ideal_components``: the
    group settings are read from the fields recorded for the owning parameter,
    never assumed.
    """
    lr = float(archive["param/group_lr"][index])
    weight_decay = float(archive["param/group_weight_decay"][index])
    eps = float(archive["param/group_eps"][index])
    bc1 = float(archive["param/bias_correction1"][index])
    bc2 = float(archive["param/bias_correction2"][index])

    p_pre = np.asarray(archive[f"{base}/param_pre"], dtype=np.float64).ravel()
    m_post = np.asarray(archive[f"{base}/exp_avg_post"], dtype=np.float64).ravel()
    v_post = np.asarray(archive[f"{base}/exp_avg_sq_post"], dtype=np.float64).ravel()

    decay = -(lr * weight_decay) * p_pre
    denom = np.sqrt(v_post) / np.sqrt(bc2) + eps
    adaptive = -(lr / bc1) * m_post / denom
    return decay + adaptive


def load_detail(archive) -> dict:
    """Concatenate every detailed coordinate in one capture.

    Returns float64 vectors for the pre-update parameter, the reconstructed
    intent, and the realized delta, plus a boolean mask marking the
    coordinates whose parameter was natively float32.
    """
    names = [str(s) for s in archive["parameter_names"]]
    roster_index = {name: i for i, name in enumerate(names)}
    bases = sorted(
        {"/".join(key.split("/")[:2]) for key in archive.files if key.startswith("layer_")}
    )
    if not bases:
        raise BundleCorrupt("capture stores no per-coordinate detail")

    p_pre, intent, delta, native_fp32, base_id, base_names = [], [], [], [], [], []
    for slot, base in enumerate(bases):
        parameter = owning_parameter(base, names)
        index = roster_index[parameter]
        if int(archive["param_meta/amsgrad"][index]) != 0:
            raise BundleCorrupt(
                f"{base}: amsgrad is on and max_exp_avg_sq is not stored, so the "
                "adaptive term cannot be reconstructed"
            )
        for field in DETAIL_FIELDS_NEEDED:
            if f"{base}/{field}" not in archive.files:
                raise BundleCorrupt(f"{base} is missing detail field {field}")
        piece = np.asarray(archive[f"{base}/param_pre"], dtype=np.float64).ravel()
        p_pre.append(piece)
        intent.append(intended_update(archive, base, index))
        delta.append(np.asarray(archive[f"{base}/delta"], dtype=np.float64).ravel())
        native_fp32.append(
            np.full(piece.size, parameter.endswith(FP32_SUFFIXES), dtype=bool)
        )
        base_id.append(np.full(piece.size, slot, dtype=np.int32))
        base_names.append(base)

    return {
        "p_pre": np.concatenate(p_pre),
        "intent": np.concatenate(intent),
        "delta": np.concatenate(delta),
        "native_fp32": np.concatenate(native_fp32),
        "base_id": np.concatenate(base_id),
        "base_names": base_names,
    }


def pooled(intent: np.ndarray, realized: np.ndarray) -> dict:
    """Pooled delivery / cosine / overshoot, with ``_pair``'s definitions.

    ``_pair`` sums squared per-tensor norms and per-tensor inner products, so
    pooling coordinate-wise over the same set of coordinates is the same
    quantity.  The identity is asserted in ``test_replay.py``.
    """
    ideal_norm = math.sqrt(float(np.sum(intent**2)))
    realized_norm = math.sqrt(float(np.sum(realized**2)))
    dot = float(np.sum(intent * realized))
    if ideal_norm == 0.0 or realized_norm == 0.0:
        return {"ideal": ideal_norm, "realized": realized_norm,
                "delivery": float("nan"), "cosine": float("nan"),
                "overshoot": float("nan"), "n_coords": int(intent.size),
                "zero_fraction": float(np.mean(realized == 0.0))}
    delivery = realized_norm / ideal_norm
    cosine = dot / (ideal_norm * realized_norm)
    return {
        "ideal": ideal_norm,
        "realized": realized_norm,
        "inner": dot,
        "delivery": delivery,
        "cosine": cosine,
        "overshoot": delivery / cosine if cosine != 0.0 else float("nan"),
        "residual_norm": math.sqrt(float(np.sum((realized - intent) ** 2))),
        "zero_fraction": float(np.mean(realized == 0.0)),
        "n_coords": int(intent.size),
    }


def apply_writer(writer, p_pre: np.ndarray, intent: np.ndarray,
                 project=None) -> np.ndarray:
    """Realized move when the parameter is written through ``writer``.

    The starting point is projected onto the writer's own lattice with
    round-to-nearest-even, the update is added in float32 (the precision the
    observed pipeline used), and the result is stored back through the writer.
    The projection is deterministic on purpose: only the *store* is supposed to
    vary between writers, so a stochastic or truncating rule must not also
    perturb the state it starts from.  For a parameter in its native format the
    projection is the identity, so the fidelity check is untouched by it.
    """
    start = (project or writer)(p_pre)
    return writer(fp32(start + intent)) - start


def writer_start(name: str, writer, p_pre: np.ndarray) -> np.ndarray:
    return PROJECTIONS.get(name, writer)(p_pre)


def native_writer_realization(data: dict, writers: dict) -> np.ndarray:
    """Realized move under each coordinate's OWN storage format.

    bfloat16 everywhere except ``.mixer.D`` and ``.mixer.dt_bias``, which the
    model holds in float32.  This is the prediction the fidelity check tests
    against the delta the run actually recorded.
    """
    realized = np.empty_like(data["p_pre"])
    is_fp32 = data["native_fp32"]
    for mask, name in ((~is_fp32, "bf16_nearest_even"), (is_fp32, "fp32")):
        if mask.any():
            realized[mask] = apply_writer(
                writers[name], data["p_pre"][mask], data["intent"][mask],
                project=PROJECTIONS[name],
            )
    return realized


def whole_model_from_scalars(archive) -> dict:
    """Whole-model pooled numbers straight from the stored per-parameter scalars.

    This is the published statistic: it covers all 146 parameters, including
    the tied embedding, not just the detailed subset.
    """
    names = [str(s) for s in archive["parameter_names"]]
    ideal_l2 = archive["param/ideal_total_l2"]
    delta_l2 = archive["param/delta_l2"]
    inner = archive["param/delivery_inner_product"]
    is_fp32 = np.array([n.endswith(FP32_SUFFIXES) for n in names])
    numel = archive["param_meta/numel"].astype(np.float64)
    zero_fraction = archive["param/delta_fraction_zero"].astype(np.float64)

    def weighted_zero(mask):
        if not mask.any():
            return float("nan")
        return float(np.sum(zero_fraction[mask] * numel[mask]) / np.sum(numel[mask]))

    out = {}
    for label, mask in (("whole", None), ("bf16", ~is_fp32), ("fp32", is_fp32)):
        stats = _pair(ideal_l2, delta_l2, inner, mask)
        stats["zero_fraction"] = weighted_zero(
            np.ones(len(names), dtype=bool) if mask is None else mask
        )
        stats["n_params"] = int(len(names) if mask is None else int(mask.sum()))
        stats["numel"] = int(numel.sum() if mask is None else numel[mask].sum())
        out[label] = stats
    return out


LOG_X_EDGES = np.linspace(-9.0, 4.0, 131)
LINEAR_X_EDGES = np.linspace(0.0, 1.0, 21)
STEP_EDGES = np.arange(-0.5, 12.5, 1.0)

# Under x ~ Uniform(0, 1) and round-to-nearest, E|r|^2 = (1/2) ULP^2 and
# E<i, r> = (3/8) ULP^2, so the overshoot is exactly 4/3.  This is the value
# the published derivation predicts, and the reference every measured
# distribution is compared against.
UNIFORM_X_OVERSHOOT = 4.0 / 3.0


def lattice_geometry(p_pre: np.ndarray, intent: np.ndarray,
                     realized: np.ndarray, ulp_fn, project) -> dict:
    """Where the intended update sits relative to one storage lattice.

    ``x = |i| / ULP(p_pre)`` is the quantity the 4/3 derivation is about, and
    ``n = r / ULP`` is how far the parameter actually moved in lattice steps.
    In these units the pooled ratio becomes::

        overshoot = sum(s^2 n^2) / sum(s^2 n x sign_match)

    where ``s`` is that coordinate's ULP.  The ULP factors cancel within a
    coordinate but not across them, so the ratio is a functional of the joint
    distribution of (x, s), weighted by ULP squared: coordinates on
    large-magnitude parameters count for more.  Both the weighted form, which
    reproduces the measured ratio exactly, and the plain unweighted form,
    which describes the shape of x on its own, are reported.

    ``ratio_predicted_from_x`` uses only ``round(x)`` in place of the realized
    move, so it can be computed from the intent without ever performing the
    write.  Agreement with the measured ratio is the test of whether
    round-to-nearest on the storage lattice is what produces the number.
    """
    start = project(p_pre)
    ulp = ulp_fn(start)
    x = np.abs(intent) / ulp
    signed_steps = np.rint(realized / ulp)
    steps = np.abs(signed_steps).astype(np.int64)
    aligned = np.sign(signed_steps) * np.sign(intent)

    positive = x > 0
    log_x = np.log10(x[positive])
    inside = x[positive & (x < 1.0)]
    linear_counts, _ = np.histogram(inside, bins=LINEAR_X_EDGES)
    expected = inside.size / (LINEAR_X_EDGES.size - 1)
    chi_square = (
        float(np.sum((linear_counts - expected) ** 2 / expected))
        if expected else float("nan")
    )

    weight = ulp**2

    def ratio(numer, denom):
        num, den = float(np.sum(numer)), float(np.sum(denom))
        return num / den if den else float("nan")

    # Predict the pooled ratio from the x distribution alone, by assuming
    # round-to-nearest picks the closest lattice point: |n| = round(x).
    predicted_steps = np.rint(x)
    predicted = ratio(weight * predicted_steps**2, weight * predicted_steps * x)
    predicted_unweighted = ratio(predicted_steps**2, predicted_steps * x)

    # Same ratio from the realized move, which is what the writer actually did.
    steps_f = steps.astype(np.float64)
    realized_ratio = ratio(weight * steps_f**2, weight * steps_f * x * aligned)
    realized_unweighted = ratio(steps_f**2, steps_f * x * aligned)
    realized_num = float(np.sum(weight * steps_f**2))

    moved = steps > 0
    single = steps == 1
    multi = steps >= 2
    above_half = x > 0.5
    return {
        "n_coords": int(x.size),
        "n_zero_intent": int(np.sum(~positive)),
        "x_below_one_fraction": float(np.mean(x < 1.0)),
        "x_mean": float(np.mean(x)),
        "x_quantiles": {
            str(q): float(np.quantile(x[positive], q))
            for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
        },
        "x_log10_decade_span_5_95": float(
            np.quantile(log_x, 0.95) - np.quantile(log_x, 0.05)
        ) if positive.any() else float("nan"),
        "x_log10_hist": {
            "edges": LOG_X_EDGES.tolist(),
            "counts": np.histogram(log_x, bins=LOG_X_EDGES)[0].tolist(),
        },
        "x_in_unit_interval_hist": {
            "edges": LINEAR_X_EDGES.tolist(),
            "counts": linear_counts.tolist(),
            "n": int(inside.size),
            "chi_square_vs_uniform": chi_square,
            "degrees_of_freedom": int(LINEAR_X_EDGES.size - 2),
            "max_bin_over_mean": float(linear_counts.max() / expected) if expected else float("nan"),
            "min_bin_over_mean": float(linear_counts.min() / expected) if expected else float("nan"),
        },
        "step_hist": {
            "edges": STEP_EDGES.tolist(),
            "counts": np.histogram(steps, bins=STEP_EDGES)[0].tolist(),
            "above_edges": int(np.sum(steps >= STEP_EDGES[-1])),
        },
        "zero_step_fraction": float(np.mean(steps == 0)),
        "abs_step_ge_2_share_of_all": float(np.mean(multi)),
        "abs_step_ge_2_share_of_moved": float(np.mean(steps[moved] >= 2)) if moved.any() else float("nan"),
        "mean_abs_step_of_moved": float(np.mean(steps[moved])) if moved.any() else float("nan"),
        # 1 / E[x | x > 1/2] is the whole ratio when every move is one step.
        # Weighted by ULP squared, to match how the pooled ratio adds up.
        "mean_x_above_half": (
            ratio(weight[above_half] * x[above_half], weight[above_half])
            if above_half.any() else float("nan")
        ),
        "inverse_mean_x_above_half": (
            1.0 / ratio(weight[above_half] * x[above_half], weight[above_half])
            if above_half.any() else float("nan")
        ),
        "single_step_only_ratio": (
            ratio(weight[single], weight[single] * x[single])
            if single.any() else float("nan")
        ),
        "multi_step_share_of_numerator": (
            float(np.sum(weight[multi] * steps_f[multi] ** 2)) / realized_num
            if realized_num else float("nan")
        ),
        "binade_edge_fraction": float(np.mean(np.frexp(start)[0] ** 2 == 0.25)),
        "ratio_from_realized_steps": realized_ratio,
        "ratio_from_realized_steps_unweighted": realized_unweighted,
        "ratio_predicted_from_x": predicted,
        "ratio_predicted_from_x_unweighted": predicted_unweighted,
        "uniform_x_reference": UNIFORM_X_OVERSHOOT,
    }


def rounding_geometry(data: dict, realized_native: np.ndarray,
                      writers: dict) -> dict:
    """Lattice geometry for both round-to-nearest formats.

    bfloat16 is the format the run actually used; float16 receives the same
    intent on a finer lattice.  If the pooled ratio is a functional of the x
    distribution rather than a constant of bfloat16, then the float16 lattice
    must produce a different x distribution and a matching different ratio.
    Restricted to the coordinates whose parameter is natively bfloat16, so the
    24 float32 tensors cannot contaminate the comparison.
    """
    bf16 = ~data["native_fp32"]
    p_pre = data["p_pre"][bf16]
    intent = data["intent"][bf16]
    out = {}
    for name in ("bf16_nearest_even", "fp16_nearest_even"):
        realized = (
            realized_native[bf16]
            if name == "bf16_nearest_even"
            else apply_writer(writers[name], p_pre, intent, PROJECTIONS[name])
        )
        out[name] = lattice_geometry(
            p_pre, intent, realized, ULP[name], PROJECTIONS[name]
        )
    return out


def replay_capture(npz_path: Path, seed: int = 42, bootstrap: int = 0) -> dict:
    """Every per-writer number for one step bundle."""
    archive = np.load(npz_path)
    data = load_detail(archive)
    writers = writer_table(seed)

    subsets = {
        "detail_all": np.ones(data["p_pre"].size, dtype=bool),
        "detail_bf16": ~data["native_fp32"],
        "detail_fp32": data["native_fp32"],
    }

    observed = {
        name: pooled(data["intent"][mask], data["delta"][mask])
        for name, mask in subsets.items()
        if mask.any()
    }

    per_writer = {}
    for name, writer in writers.items():
        realized = apply_writer(
            writer, data["p_pre"], data["intent"], PROJECTIONS[name]
        )
        per_writer[name] = {
            subset: pooled(data["intent"][mask], realized[mask])
            for subset, mask in subsets.items()
            if mask.any()
        }

    realized_native = native_writer_realization(data, writers)
    match = realized_native == data["delta"]
    bf16 = ~data["native_fp32"]
    native_pooled = {
        subset: pooled(data["intent"][mask], realized_native[mask])
        for subset, mask in subsets.items()
        if mask.any()
    }
    scale = np.maximum(np.abs(data["delta"]), np.abs(realized_native))
    mismatch = ~match
    fidelity = {
        "match_rate_all": float(np.mean(match)),
        "match_rate_bf16": float(np.mean(match[bf16])) if bf16.any() else float("nan"),
        "match_rate_fp32": float(np.mean(match[~bf16])) if (~bf16).any() else float("nan"),
        "n_mismatch": int(np.sum(mismatch)),
        "max_mismatch_in_ulp": float(
            np.max(
                np.abs(realized_native[mismatch] - data["delta"][mismatch])
                / bf16_ulp(bf16_nearest_even(data["p_pre"][mismatch]))
            )
        ) if mismatch.any() else 0.0,
        "max_relative_mismatch": float(
            np.max(
                np.abs(realized_native[mismatch] - data["delta"][mismatch])
                / np.where(scale[mismatch] > 0, scale[mismatch], 1.0)
            )
        ) if mismatch.any() else 0.0,
        "overshoot_native_replay": native_pooled["detail_all"]["overshoot"],
        "overshoot_observed": observed["detail_all"]["overshoot"],
        "overshoot_abs_gap": abs(
            native_pooled["detail_all"]["overshoot"] - observed["detail_all"]["overshoot"]
        ),
    }

    record = {
        "optimizer_step": int(archive["optimizer_step"]),
        "dataset_row": int(archive["dataset_row"]),
        "training_loss": float(archive["training_loss"]),
        "clip_coefficient": float(archive["clip/clip_coefficient_empirical"]),
        "clip_fired": int(archive["clip/clip_fired"]),
        "grad_norm_preclip": float(archive["clip/observer_pre_clip_total_norm"]),
        "lr": float(archive["param/group_lr"][0]),
        "n_detail_coords": int(data["p_pre"].size),
        "n_detail_bf16": int(np.sum(bf16)),
        "n_detail_fp32": int(np.sum(~bf16)),
        "n_detail_bases": len(data["base_names"]),
        "detail_bases": data["base_names"],
        "observed": observed,
        "native_replay": native_pooled,
        "writers": per_writer,
        "fidelity": fidelity,
        "geometry": rounding_geometry(data, realized_native, writers),
        "whole_model_scalars": whole_model_from_scalars(archive),
    }

    if bootstrap:
        # Over the bfloat16 detail coordinates, which are the headline subset.
        record["bootstrap"] = _bootstrap_overshoot(
            data, writers, subsets["detail_bf16"], bootstrap, seed
        )
        record["bootstrap_subset"] = "detail_bf16"
    return record


def _bootstrap_overshoot(data, writers, mask, draws: int, seed: int) -> dict:
    """Resample coordinates to show how tightly the pooled ratio is pinned.

    The detail subset is the whole population of retained coordinates, so this
    is not sampling error in the usual sense.  It answers a narrower question:
    how much of the writer separation could be an accident of which
    coordinates the capture happened to keep.
    """
    rng = np.random.default_rng(seed)
    p_pre = data["p_pre"][mask]
    intent = data["intent"][mask]
    out = {}
    realized_by_writer = {
        name: apply_writer(writer, p_pre, intent, PROJECTIONS[name])
        for name, writer in writers.items()
    }
    n = intent.size
    for name, realized in realized_by_writer.items():
        values = []
        for _ in range(draws):
            idx = rng.integers(0, n, size=n)
            i_b, r_b = intent[idx], realized[idx]
            dot = float(np.sum(i_b * r_b))
            values.append(float(np.sum(r_b**2)) / dot if dot else float("nan"))
        values = np.array(values)
        out[name] = {
            "lo": float(np.quantile(values, 0.025)),
            "hi": float(np.quantile(values, 0.975)),
            "draws": draws,
        }
    return out
