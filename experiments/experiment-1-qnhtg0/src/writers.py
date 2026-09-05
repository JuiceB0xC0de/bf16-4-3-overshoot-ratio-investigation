"""Parameter writers: exact float64 -> lattice rounding, one function per rule.

Every writer takes a float64 array of *exact* target values and returns the
float64 value that the corresponding storage format would actually hold.  The
point of Arm A is to hold the intended update fixed and swap only this
function, so the writers must differ in nothing but the rounding rule.

bfloat16 is not a numpy dtype, so it is implemented by scaling.  For a finite
``x`` in the binade ``[2^(e-1), 2^e)`` the bfloat16 lattice spacing is
``s = 2^(e-8)`` (one sign bit, eight exponent bits, seven stored mantissa bits
plus the implicit leading one).  ``x / s`` is then exactly representable in
float64 and lands in ``[128, 256)``, so rounding it to an integer and
multiplying back is exact: no double rounding, unlike a float32 intermediate.
Subnormal bfloat16 is the clamp at ``s = 2^-133``.
"""

from __future__ import annotations

import numpy as np

# Smallest bfloat16 subnormal spacing: min normal 2^-126 with 7 mantissa bits.
BF16_MIN_ULP_EXP = -133
BF16_MIN_ULP = float(np.ldexp(1.0, BF16_MIN_ULP_EXP))


def bf16_ulp(x: np.ndarray) -> np.ndarray:
    """Spacing of the bfloat16 lattice at ``x`` (float64 in, float64 out)."""
    x = np.asarray(x, dtype=np.float64)
    _, exponent = np.frexp(x)  # x == mantissa * 2**exponent, |mantissa| in [0.5, 1)
    spacing = np.ldexp(1.0, np.maximum(exponent - 8, BF16_MIN_ULP_EXP))
    return np.where(x == 0.0, BF16_MIN_ULP, spacing)


def _bf16_scaled(x: np.ndarray, quantize) -> np.ndarray:
    spacing = bf16_ulp(x)
    return spacing * quantize(x / spacing)


def bf16_nearest_even(x: np.ndarray) -> np.ndarray:
    """IEEE round-to-nearest, ties-to-even: what a bf16 store actually does."""
    return _bf16_scaled(x, np.rint)


def bf16_truncate(x: np.ndarray) -> np.ndarray:
    """Round toward zero: the rule that loses magnitude instead of snapping."""
    return _bf16_scaled(x, np.trunc)


def make_bf16_stochastic(seed: int):
    """Unbiased stochastic rounding, seeded so a rerun reproduces byte-exactly."""
    rng = np.random.default_rng(seed)

    def bf16_stochastic(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        spacing = bf16_ulp(x)
        scaled = x / spacing
        return spacing * np.floor(scaled + rng.random(scaled.shape))

    return bf16_stochastic


# float16: ten stored mantissa bits, min normal 2^-14, min subnormal 2^-24.
FP16_MIN_ULP_EXP = -24


def fp16_ulp(x: np.ndarray) -> np.ndarray:
    """Spacing of the float16 lattice at ``x``."""
    x = np.asarray(x, dtype=np.float64)
    _, exponent = np.frexp(x)
    spacing = np.ldexp(1.0, np.maximum(exponent - 11, FP16_MIN_ULP_EXP))
    return np.where(x == 0.0, np.ldexp(1.0, FP16_MIN_ULP_EXP), spacing)


def fp16_nearest_even(x: np.ndarray) -> np.ndarray:
    """float16 round-to-nearest-even, including its narrower exponent range."""
    return np.asarray(x, dtype=np.float64).astype(np.float16).astype(np.float64)


def fp32(x: np.ndarray) -> np.ndarray:
    """float32 round-to-nearest-even: the negative control."""
    return np.asarray(x, dtype=np.float64).astype(np.float32).astype(np.float64)


def writer_table(seed: int = 42) -> dict:
    """Name -> writer.  Order is the order they appear in every output table."""
    return {
        "bf16_nearest_even": bf16_nearest_even,
        "bf16_stochastic": make_bf16_stochastic(seed),
        "bf16_truncate": bf16_truncate,
        "fp16_nearest_even": fp16_nearest_even,
        "fp32": fp32,
    }


# The deterministic projection onto each writer's lattice.  A writer's starting
# point must be a state its own format can hold, but the projection must not
# itself be the thing under test, so every writer starts from the
# round-to-nearest-even image of the recorded parameter.
PROJECTIONS = {
    "bf16_nearest_even": bf16_nearest_even,
    "bf16_stochastic": bf16_nearest_even,
    "bf16_truncate": bf16_nearest_even,
    "fp16_nearest_even": fp16_nearest_even,
    "fp32": fp32,
}

# Lattice spacing for the writers whose rule is round-to-nearest, used to
# express an intended update in units of the storage resolution.
ULP = {
    "bf16_nearest_even": bf16_ulp,
    "bf16_stochastic": bf16_ulp,
    "bf16_truncate": bf16_ulp,
    "fp16_nearest_even": fp16_ulp,
}
