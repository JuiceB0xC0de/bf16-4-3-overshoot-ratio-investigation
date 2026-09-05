"""Checks on the replay machinery, including a case with a known answer.

The important one is ``test_uniform_x_gives_four_thirds``: it feeds the real
pipeline a synthetic update whose sizes are uniform on (0, 1) ULP, which is
exactly the assumption the published derivation makes, and requires the
measured pooled ratio to come out at 4/3.  If the machinery could not
reproduce 4/3 where 4/3 is provably the right answer, nothing it says about
the real captures would be readable.
"""

from __future__ import annotations

import math

import numpy as np

from replay import (
    UNIFORM_X_OVERSHOOT,
    BundleCorrupt,
    apply_writer,
    lattice_geometry,
    owning_parameter,
    pooled,
)
from transmission_extract import _pair
from writers import PROJECTIONS, ULP, bf16_nearest_even, bf16_ulp, writer_table


def _synthetic_parameters(n: int, seed: int) -> np.ndarray:
    """Parameters in the middle of their binade, so the closed forms are exact.

    A bfloat16 value sitting exactly on a power of two has a finer lattice on
    its inward side than its outward side, so an update of a fixed size in ULP
    rounds differently depending on direction.  That asymmetry is real and is
    measured on the captures, but it would make the analytic reference for
    these tests ambiguous, so the synthetic parameters stay away from binade
    edges by more than the largest update the tests apply.
    """
    rng = np.random.default_rng(seed)
    magnitude = np.ldexp(1.25 + 0.5 * rng.random(n), rng.integers(-12, 2, size=n))
    return bf16_nearest_even(magnitude * rng.choice([-1.0, 1.0], size=n))


def test_pooled_matches_pair() -> None:
    """Coordinate-wise pooling is the same quantity ``_pair`` computes."""
    rng = np.random.default_rng(3)
    tensors = [
        (rng.normal(size=size), rng.normal(size=size))
        for size in (40, 7, 1000, 3)
    ]
    ideal_l2 = np.array([np.linalg.norm(i) for i, _ in tensors])
    realized_l2 = np.array([np.linalg.norm(r) for _, r in tensors])
    inner = np.array([float(np.dot(i, r)) for i, r in tensors])
    reference = _pair(ideal_l2, realized_l2, inner)

    flat_i = np.concatenate([i for i, _ in tensors])
    flat_r = np.concatenate([r for _, r in tensors])
    mine = pooled(flat_i, flat_r)
    for key in ("delivery", "cosine", "overshoot"):
        assert math.isclose(mine[key], reference[key], rel_tol=1e-12), key


def test_uniform_x_gives_four_thirds() -> None:
    """The derivation's own assumption, run through the real pipeline."""
    n = 4_000_000
    rng = np.random.default_rng(11)
    p_pre = _synthetic_parameters(n, seed=5)
    spacing = bf16_ulp(p_pre)
    # |intent| uniform on (0, 1) ULP, sign independent of the parameter.
    intent = rng.random(n) * spacing * rng.choice([-1.0, 1.0], size=n)

    writers = writer_table(42)
    realized = apply_writer(
        writers["bf16_nearest_even"], p_pre, intent, PROJECTIONS["bf16_nearest_even"]
    )
    measured = pooled(intent, realized)["overshoot"]
    # 2e6 effective moves; the pooled ratio concentrates well inside 0.005.
    assert abs(measured - UNIFORM_X_OVERSHOOT) < 5e-3, (
        f"uniform x gave overshoot {measured:.5f}, expected {UNIFORM_X_OVERSHOOT:.5f}"
    )

    geometry = lattice_geometry(
        p_pre, intent, realized, ULP["bf16_nearest_even"],
        PROJECTIONS["bf16_nearest_even"],
    )
    assert abs(geometry["ratio_predicted_from_x"] - UNIFORM_X_OVERSHOOT) < 5e-3
    assert abs(geometry["ratio_from_realized_steps"] - measured) < 1e-9
    assert abs(geometry["inverse_mean_x_above_half"] - UNIFORM_X_OVERSHOOT) < 5e-3
    assert abs(geometry["x_in_unit_interval_hist"]["max_bin_over_mean"] - 1.0) < 0.02


def test_doubling_the_updates_moves_the_ratio() -> None:
    """Doubling every update, so x ~ U(0, 2), must change the ratio.

    Under x ~ U(0, 2) round-to-nearest gives n = 0, 1, 2 with probability
    1/4, 1/2, 1/4, so E[n^2] = 3/2 and E[n x] = 11/8 and the pooled ratio is
    12/11 = 1.0909.  A pipeline that returned 4/3 here would be reporting a
    constant of bfloat16 rather than measuring the update distribution.
    """
    n = 4_000_000
    rng = np.random.default_rng(12)
    p_pre = _synthetic_parameters(n, seed=6)
    spacing = bf16_ulp(p_pre)
    intent = 2.0 * rng.random(n) * spacing * rng.choice([-1.0, 1.0], size=n)
    realized = apply_writer(
        writer_table(42)["bf16_nearest_even"], p_pre, intent,
        PROJECTIONS["bf16_nearest_even"],
    )
    measured = pooled(intent, realized)["overshoot"]
    assert abs(measured - 12.0 / 11.0) < 5e-3, f"got {measured:.5f}, expected 1.09091"


def test_fp32_writer_is_near_lossless() -> None:
    n = 500_000
    rng = np.random.default_rng(13)
    p_pre = _synthetic_parameters(n, seed=7)
    intent = rng.normal(scale=1e-4, size=n)
    realized = apply_writer(writer_table(42)["fp32"], p_pre, intent, PROJECTIONS["fp32"])
    assert abs(pooled(intent, realized)["overshoot"] - 1.0) < 1e-3


def _roster(n_layers: int = 12) -> list[str]:
    names = ["backbone.embedding.weight", "backbone.norm_f.weight"]
    for layer in range(n_layers):
        stem = f"backbone.layers.{layer}"
        names += [
            f"{stem}.norm.weight",
            f"{stem}.mixer.dt_bias",
            f"{stem}.mixer.B_bias",
            f"{stem}.mixer.D",
            f"{stem}.mixer.in_proj.weight",
            f"{stem}.mixer.out_proj.weight",
            f"{stem}.mixer.norm.weight",
        ]
    return names


def test_owning_parameter() -> None:
    names = _roster()
    assert owning_parameter("layer_07/angles", names) == \
        "backbone.layers.7.mixer.in_proj.weight"
    assert owning_parameter("layer_00/dd_A", names) == \
        "backbone.layers.0.mixer.in_proj.weight"
    assert owning_parameter("layer_11/dt_bias", names) == \
        "backbone.layers.11.mixer.dt_bias"
    assert owning_parameter("layer_03/D", names) == "backbone.layers.3.mixer.D"

    # A layer with two parameters whose names both end in norm.weight must
    # raise rather than pick one: the wrong owner would supply the wrong
    # learning rate and weight decay to the reconstruction.
    try:
        owning_parameter("layer_03/norm_weight", names)
    except BundleCorrupt:
        pass
    else:
        raise AssertionError("ambiguous owner was resolved silently")

    try:
        owning_parameter("layer_03/dd_A", [n for n in names if "in_proj" not in n])
    except BundleCorrupt:
        pass
    else:
        raise AssertionError("missing owner was resolved silently")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
