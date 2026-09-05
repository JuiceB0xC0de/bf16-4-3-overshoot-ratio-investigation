"""The torch geometry port must agree with Arm A's numpy original.

Arm A (exp_01m1rbjk2pe84tfg8xj7qnhtg0) validated its numpy implementation
against torch's own bf16 casts and against closed forms, and its replay
reproduced the recorded pooled overshoot to 0.00048.  This file holds the
torch port to that same standard: identical ULP lattice, identical pooled
rule, identical lattice-geometry fields, on data whose answer is known.

Run: ``apy src/test_geometry.py``  (CPU only, seconds).
"""

from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reference"))

import arm_a_writers as aw            # noqa: E402  Arm A, verbatim

# Arm A's replay imports its writers by bare module name; alias rather than
# keep a second copy of the same file in this tree.
sys.modules.setdefault("writers", aw)

# transmission_extract imports orjson only for its CLI; stub it so the
# project's own pooling rule can be imported rather than restated.
try:
    import orjson  # noqa: F401
except ModuleNotFoundError:
    sys.modules["orjson"] = types.ModuleType("orjson")

from geometry import (                # noqa: E402
    LatticeAccumulator, PooledAccumulator, assert_on_lattice, bf16_ulp,
)

FAILURES = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def sample_bf16(n: int, seed: int) -> torch.Tensor:
    """Values spanning the magnitudes a real parameter tensor covers."""
    g = torch.Generator().manual_seed(seed)
    mag = torch.exp(torch.empty(n).uniform_(-12.0, 2.0, generator=g))
    sign = torch.where(torch.rand(n, generator=g) < 0.5, -1.0, 1.0)
    x = (mag * sign).to(torch.bfloat16).to(torch.float32)
    x[: n // 50] = 0.0
    return x


def test_ulp_matches_arm_a():
    x = sample_bf16(200_000, 0)
    mine = bf16_ulp(x).double().numpy()
    theirs = aw.bf16_ulp(x.double().numpy())
    check("bf16_ulp == Arm A bf16_ulp", bool(np.array_equal(mine, theirs)),
          f"max rel diff {np.max(np.abs(mine - theirs) / theirs):.3e}")


def test_ulp_is_the_real_bf16_spacing():
    """The next bf16 value up must be exactly one ULP away."""
    x = sample_bf16(50_000, 1)
    x = x[x > 0]
    up = torch.nextafter(x.to(torch.bfloat16), torch.tensor(float("inf"),
                                                            dtype=torch.bfloat16))
    spacing = (up.to(torch.float32) - x)
    check("bf16_ulp == nextafter spacing",
          bool(torch.equal(spacing, bf16_ulp(x))))


def test_on_lattice():
    x = sample_bf16(1000, 2)
    ok = True
    try:
        assert_on_lattice(x)
    except AssertionError:
        ok = False
    off = x.clone()
    off[5] = 0.1234567
    raised = False
    try:
        assert_on_lattice(off)
    except AssertionError:
        raised = True
    check("assert_on_lattice accepts bf16 values and rejects others", ok and raised)


def test_pooled_matches_pair_rule():
    """Coordinate-wise pooling == the project's own per-tensor ``_pair``.

    ``_pair`` pools by summing squared per-tensor norms and per-tensor inner
    products.  Accumulating coordinate-wise over the same coordinates has to
    give the same number, and it is the project's own function that is called
    here rather than a restatement of it.
    """
    from transmission_extract import _pair  # noqa: PLC0415

    g = torch.Generator().manual_seed(3)
    # A realistic geometry: realized is a rounded version of intent, so the
    # inner product is well conditioned rather than a near-cancellation.
    tensors = []
    for n in (997, 4096, 15_001):
        intent = torch.randn(n, generator=g, dtype=torch.float64) * 1e-4
        realized = (intent + torch.randn(n, generator=g, dtype=torch.float64) * 3e-5)
        realized[realized.abs() < 4e-5] = 0.0
        tensors.append((intent, realized))

    acc = PooledAccumulator()
    for intent, realized in tensors:
        acc.add(intent, realized)
    mine = acc.result()

    ideal_l2 = np.array([float(torch.linalg.vector_norm(i)) for i, _ in tensors])
    delta_l2 = np.array([float(torch.linalg.vector_norm(r)) for _, r in tensors])
    inner = np.array([float(torch.sum(i * r)) for i, r in tensors])
    theirs = _pair(ideal_l2, delta_l2, inner)

    rel = abs(mine["overshoot"] - theirs["overshoot"]) / abs(theirs["overshoot"])
    check("pooled overshoot == transmission_extract._pair", rel < 1e-12,
          f"{mine['overshoot']:.12f} vs {theirs['overshoot']:.12f} (rel {rel:.2e})")
    rel_d = abs(mine["delivery"] - theirs["delivery"]) / abs(theirs["delivery"])
    check("pooled delivery == transmission_extract._pair", rel_d < 1e-12)


def _arm_a_geometry(p_pre, intent, realized):
    import arm_a_replay  # noqa: PLC0415

    return arm_a_replay.lattice_geometry(
        p_pre.double().numpy(), intent.double().numpy(),
        realized.double().numpy(), aw.bf16_ulp, aw.bf16_nearest_even)


def test_lattice_matches_arm_a():
    """Same fields, same numbers, on a realistic (p, intent, realized) triple."""
    g = torch.Generator().manual_seed(4)
    p_pre = sample_bf16(400_000, 5)
    p_pre = p_pre[p_pre != 0]
    ulp = bf16_ulp(p_pre)
    # Intent spread over ~4 decades of x, as in the real data.
    x = torch.exp(torch.empty(p_pre.numel()).uniform_(-6.0, 2.0, generator=g))
    intent = x * ulp * torch.where(torch.rand(p_pre.numel(), generator=g) < 0.5, -1.0, 1.0)
    realized = ((p_pre + intent).to(torch.bfloat16).to(torch.float32) - p_pre)

    acc = LatticeAccumulator(torch.device("cpu"))
    acc.add(p_pre, intent, realized)
    mine = acc.result()
    theirs = _arm_a_geometry(p_pre, intent, realized)

    exact = [
        "x_below_one_fraction", "x_mean", "zero_step_fraction",
        "abs_step_ge_2_share_of_all", "abs_step_ge_2_share_of_moved",
        "mean_abs_step_of_moved", "mean_x_above_half",
        "inverse_mean_x_above_half", "single_step_only_ratio",
        "multi_step_share_of_numerator", "binade_edge_fraction",
        "ratio_from_realized_steps", "ratio_from_realized_steps_unweighted",
        "ratio_predicted_from_x", "ratio_predicted_from_x_unweighted",
    ]
    worst, worst_name = 0.0, ""
    for key in exact:
        a, b = mine[key], theirs[key]
        rel = abs(a - b) / max(abs(b), 1e-30)
        if rel > worst:
            worst, worst_name = rel, key
    check("lattice_geometry scalars == Arm A", worst < 1e-9,
          f"worst rel diff {worst:.3e} on {worst_name}")

    med_mine = mine["x_quantiles_from_hist"]["0.5"]
    med_theirs = theirs["x_quantiles"]["0.5"]
    check("x median from histogram ~ Arm A exact median",
          abs(med_mine - med_theirs) / med_theirs < 0.02,
          f"{med_mine:.6f} vs {med_theirs:.6f}")

    check("step_hist counts == Arm A",
          mine["step_hist"]["counts"] == theirs["step_hist"]["counts"])
    check("unit-interval hist counts == Arm A",
          mine["x_in_unit_interval_hist"]["counts"]
          == theirs["x_in_unit_interval_hist"]["counts"])


def test_uniform_x_closed_form():
    """x ~ U(0,1) on a constant lattice gives exactly the 4/3 idealisation.

    This is the number the finding is named after.  It is a property of a
    uniform x, not of bfloat16 -- Arm A measured the real x distribution and
    found it nothing like uniform, so this test exists to show the code
    reproduces the idealisation, not to suggest the data obeys it.
    """
    n = 4_000_000
    g = torch.Generator().manual_seed(6)
    p_pre = torch.full((n,), 1.0)                     # constant ULP => unweighted
    ulp = bf16_ulp(p_pre)
    x = torch.rand(n, generator=g)
    intent = x * ulp
    realized = torch.round(x) * ulp
    acc = LatticeAccumulator(torch.device("cpu"))
    acc.add(p_pre, intent, realized)
    got = acc.result()["ratio_from_realized_steps"]
    check("uniform x on a flat lattice -> 4/3", abs(got - 4 / 3) < 2e-3,
          f"{got:.6f} vs {4/3:.6f}")


def test_accumulation_is_order_invariant():
    """Splitting a tensor across add() calls must not change any total."""
    p_pre = sample_bf16(120_000, 7)
    p_pre = p_pre[p_pre != 0]
    ulp = bf16_ulp(p_pre)
    g = torch.Generator().manual_seed(8)
    intent = torch.exp(torch.empty(p_pre.numel()).uniform_(-5.0, 1.0, generator=g)) * ulp
    realized = ((p_pre + intent).to(torch.bfloat16).to(torch.float32) - p_pre)

    whole = LatticeAccumulator(torch.device("cpu"))
    whole.add(p_pre, intent, realized)
    split = LatticeAccumulator(torch.device("cpu"))
    for sl in (slice(0, 5000), slice(5000, 60_000), slice(60_000, None)):
        split.add(p_pre[sl], intent[sl], realized[sl])
    a, b = whole.result(), split.result()
    worst = max(abs(a[k] - b[k]) / max(abs(b[k]), 1e-30)
                for k in ("ratio_from_realized_steps", "ratio_predicted_from_x",
                          "mean_x_above_half", "zero_step_fraction"))
    check("per-parameter accumulation == single-shot", worst < 1e-12,
          f"worst rel diff {worst:.3e}")



def test_bucketize_matches_numpy_histogram():
    """torch.bucketize must bin exactly as numpy.histogram, boundaries included.

    The two libraries differ on which side of an edge is closed, and a silent
    disagreement would shift every histogram by one bin at the edges.  Checked
    on values placed deliberately ON the edges.
    """
    from geometry import LINEAR_X_EDGES  # noqa: PLC0415

    # Both sides must see the same numbers: the accumulator bins float32 x
    # against float32 edges, so numpy is given the float32 images too.  Mixing
    # float64 edge constants into one side only would manufacture off-by-one
    # counts that say nothing about the binning rule.
    edges = torch.tensor(LINEAR_X_EDGES, dtype=torch.float32)
    edges_np = edges.numpy().astype(np.float64)
    values = torch.cat([
        edges[:-1],                                               # exactly on edges
        torch.rand(50_000, generator=torch.Generator().manual_seed(9)),
    ])
    mine = torch.bincount(torch.bucketize(values, edges, right=True),
                          minlength=len(LINEAR_X_EDGES) + 1)[1:-1].numpy()
    theirs, _ = np.histogram(values.numpy().astype(np.float64), bins=edges_np)
    check("bucketize(right=True) == np.histogram binning",
          bool(np.array_equal(mine, theirs)),
          f"n_on_edge={len(edges) - 1}, total={int(mine.sum())}")


if __name__ == "__main__":
    print("geometry port vs Arm A original")
    for fn in (test_ulp_matches_arm_a, test_ulp_is_the_real_bf16_spacing,
               test_on_lattice, test_pooled_matches_pair_rule,
               test_lattice_matches_arm_a, test_uniform_x_closed_form,
               test_bucketize_matches_numpy_histogram,
               test_accumulation_is_order_invariant):
        fn()
    print(f"\n{len(FAILURES)} failure(s)" + (f": {FAILURES}" if FAILURES else ""))
    sys.exit(1 if FAILURES else 0)
