"""Check the writers against an independent implementation (torch).

The replay hands every writer a value that is exactly float32-representable,
because the observed pipeline computed the AdamW update in float32 and the
bf16 store rounds from there (``adamw_decomposition_semantics =
fp32_semantic_reconstruction``).  So the reference comparison is over float32
inputs, where torch's float32 -> bfloat16 cast is an unambiguous single
rounding.  Feeding the writers a float64 value that is not float32
representable is outside the replay's contract; torch would double-round it
through float32 and this implementation would not.
"""

from __future__ import annotations

import numpy as np

from writers import (
    bf16_nearest_even,
    bf16_truncate,
    bf16_ulp,
    fp16_nearest_even,
    fp32,
    make_bf16_stochastic,
)


def _reference_values(n: int = 400_000) -> np.ndarray:
    """float32-representable values covering the awkward places."""
    rng = np.random.default_rng(0)
    magnitudes = np.ldexp(rng.random(n), rng.integers(-140, 8, size=n))
    values = magnitudes * rng.choice([-1.0, 1.0], size=n)
    lattice = bf16_nearest_even(values)
    spacing = bf16_ulp(lattice)
    extras = np.concatenate(
        [
            lattice[: n // 4],                            # exactly on lattice
            lattice[: n // 4] + 0.5 * spacing[: n // 4],  # exact midpoints
            np.ldexp(1.0, np.arange(-140, 8, dtype=np.int32)),  # binade edges
            np.array([0.0, -0.0]),
        ]
    )
    return fp32(np.concatenate([values, extras]))


def test_bf16_matches_torch() -> None:
    import torch

    values = _reference_values()
    reference = (
        torch.from_numpy(values.astype(np.float32))
        .to(torch.bfloat16)
        .to(torch.float64)
        .numpy()
    )
    bad = int(np.sum(bf16_nearest_even(values) != reference))
    assert bad == 0, f"{bad} of {values.size} disagree with torch bfloat16"


def test_fp16_matches_torch() -> None:
    import torch

    values = _reference_values()
    reference = (
        torch.from_numpy(values.astype(np.float32))
        .to(torch.float16)
        .to(torch.float64)
        .numpy()
    )
    bad = int(np.sum(fp16_nearest_even(values) != reference))
    assert bad == 0, f"{bad} of {values.size} disagree with torch float16"


def test_truncate_never_overshoots() -> None:
    values = _reference_values()
    truncated = bf16_truncate(values)
    assert np.all(np.abs(truncated) <= np.abs(values))
    assert np.all(np.sign(truncated) * np.sign(values) >= 0)


def test_stochastic_is_unbiased() -> None:
    rng = np.random.default_rng(1)
    base = bf16_nearest_even(rng.random(2000) + 1.0)
    spacing = bf16_ulp(base)
    target = fp32(base + 0.3 * spacing)
    n_draws = 400
    writer = make_bf16_stochastic(7)
    draws = np.stack([writer(target) for _ in range(n_draws)])
    # Every draw must be one of the two bracketing lattice points.
    assert np.all(np.isin(draws, np.stack([base, base + spacing]).T.ravel()))
    # Pooled bias: 2000 * 400 Bernoulli(0.3) draws, standard error in ULP.
    err_ulp = (draws - target) / spacing
    se = np.sqrt(0.3 * 0.7 / err_ulp.size)
    assert abs(err_ulp.mean()) < 4.0 * se, (
        f"stochastic rounding biased: {err_ulp.mean():+.5f} ULP, 4 se = {4 * se:.5f}"
    )


def test_writers_are_idempotent_on_their_own_lattice() -> None:
    values = _reference_values()
    for writer in (bf16_nearest_even, bf16_truncate, fp16_nearest_even, fp32):
        once = writer(values)
        assert np.all(writer(once) == once), writer.__name__


def test_bf16_ulp_is_the_lattice_spacing() -> None:
    # Stepping is checked away from zero, which stays inside the binade whose
    # spacing bf16_ulp reports.  Toward zero the spacing halves at every
    # power of two, so a step of one ULP there is genuinely more than one
    # lattice point and the check would be about binade edges, not spacing.
    values = _reference_values()
    lattice = bf16_nearest_even(values)
    lattice = lattice[lattice != 0.0]
    spacing = bf16_ulp(lattice) * np.sign(lattice)
    normal = np.abs(lattice) >= np.ldexp(1.0, -126)
    stepped = bf16_nearest_even(fp32(lattice + spacing))
    assert np.all(stepped[normal] != lattice[normal])
    assert np.all(np.abs(stepped[normal]) > np.abs(lattice[normal]))
    inside = bf16_nearest_even(fp32(lattice + 0.4 * spacing))
    assert np.all(inside[normal] == lattice[normal])


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
