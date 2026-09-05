"""Pooling and bf16 lattice geometry, in torch, accumulated across parameters.

This is a port of Arm A's ``src/writers.py::bf16_ulp`` and
``src/replay.py::{pooled, lattice_geometry}`` (experiment
exp_01m1rbjk2pe84tfg8xj7qnhtg0) from float64 numpy over one capture's detail
coordinates to torch over every parameter of a live run.  The definitions are
unchanged; ``test_geometry.py`` asserts agreement with the numpy original to
float64 round-off on random data and on closed forms.

Two things differ from Arm A by necessity:

* Arm A held one capture's ~959k detail coordinates in memory at once.  Here
  the pooled quantities and every histogram are *accumulated* per parameter, so
  peak memory is one parameter rather than the model.  Every reported quantity
  is an exact sum or count over the same coordinate set, not a subsample -- the
  only approximation is that quantiles are read off a 0.02-decade histogram
  rather than sorted exactly, and those are labelled ``_from_hist``.
* Arm A projected the recorded parameter onto the writer's lattice before
  measuring.  Here the parameter *is* natively bfloat16, so that projection is
  the identity; ``assert_on_lattice`` checks it rather than assuming it.

The pooled ratio in lattice units, from Arm A's docstring::

    overshoot = sum(s^2 n^2) / sum(s^2 n x sign_match)

with ``s`` the coordinate's ULP, ``x = |intent|/s`` and ``n = round(r/s)``.
The ULP factors cancel within a coordinate but not across them, so the ratio is
a functional of the joint distribution of (x, s) weighted by ULP squared.

Arm A found x is emphatically NOT uniform on (0,1) in real data (chi-square
3.284e6 on 19 dof, median x 0.0757, 2.43 decades between the 5th and 95th
percentile) and that 35.88% of nonzero moves are two or more ULPs.  So
``uniform_x_reference = 4/3`` is recorded only as the idealisation it is, and
the full ``step_hist`` is retained rather than a zero/nonzero split.
"""

from __future__ import annotations

import math

import numpy as np
import torch

BF16_MIN_ULP_EXP = -133
BF16_MANTISSA_BITS = 8  # 7 stored + implicit leading one
UNIFORM_X_OVERSHOOT = 4.0 / 3.0

# Histogram edges.  LOG_X uses 0.02-decade bins (Arm A used 0.1) so quantiles
# read off the histogram are accurate to well under a percent in x.
LOG_X_EDGES = np.round(np.arange(-9.0, 4.0 + 1e-9, 0.02), 6)
LINEAR_X_EDGES = np.linspace(0.0, 1.0, 21)
STEP_EDGES = np.arange(-0.5, 12.5, 1.0)
# Realized moves are non-negative integers in ULPs, so they are counted by
# value rather than bucketized: index k is exactly "moved k ULPs", and the
# final slot is the overflow that numpy's histogram would drop off the end.
N_STEP_BINS = len(STEP_EDGES) - 1


def bf16_ulp(x: torch.Tensor) -> torch.Tensor:
    """Spacing of the bfloat16 lattice at ``x``.  Mirrors Arm A's numpy form."""
    _, exponent = torch.frexp(x)
    exponent = torch.clamp(exponent - BF16_MANTISSA_BITS, min=BF16_MIN_ULP_EXP)
    spacing = torch.ldexp(torch.ones_like(x), exponent)
    return torch.where(x == 0, torch.full_like(x, 2.0**BF16_MIN_ULP_EXP), spacing)


def assert_on_lattice(p: torch.Tensor) -> None:
    """A natively-bfloat16 parameter upcast to fp32 is already on the lattice."""
    round_trip = p.to(torch.bfloat16).to(p.dtype)
    if not torch.equal(round_trip, p):
        n = int((round_trip != p).sum())
        raise AssertionError(
            f"{n} of {p.numel()} values are not on the bf16 lattice; the "
            "geometry's projection assumption does not hold"
        )


class PooledAccumulator:
    """Sums that make up ``transmission_extract.py::_pair`` over a coord set.

    ``_pair`` sums squared per-tensor norms and per-tensor inner products, so
    accumulating coordinate-wise over the same coordinates is the same number.
    """

    def __init__(self) -> None:
        self.ideal_sq = 0.0
        self.realized_sq = 0.0
        self.inner = 0.0
        self.residual_sq = 0.0
        self.n = 0
        self.n_zero = 0
        self.n_params = 0

    def add(self, intent: torch.Tensor, realized: torch.Tensor) -> None:
        self.ideal_sq += float(torch.sum(intent * intent, dtype=torch.float64))
        self.realized_sq += float(torch.sum(realized * realized, dtype=torch.float64))
        self.inner += float(torch.sum(intent * realized, dtype=torch.float64))
        diff = realized - intent
        self.residual_sq += float(torch.sum(diff * diff, dtype=torch.float64))
        self.n += intent.numel()
        self.n_zero += int((realized == 0).sum())
        self.n_params += 1

    def result(self) -> dict:
        ideal = math.sqrt(self.ideal_sq)
        realized = math.sqrt(self.realized_sq)
        out = {
            "ideal": ideal,
            "realized": realized,
            "inner": self.inner,
            "residual_norm": math.sqrt(self.residual_sq),
            "zero_fraction": self.n_zero / self.n if self.n else float("nan"),
            "n_coords": self.n,
            "n_params": self.n_params,
        }
        if ideal == 0.0 or realized == 0.0:
            out.update(delivery=float("nan"), cosine=float("nan"),
                       overshoot=float("nan"))
            return out
        delivery = realized / ideal
        cosine = self.inner / (ideal * realized)
        out.update(
            delivery=delivery,
            cosine=cosine,
            overshoot=delivery / cosine if cosine != 0.0 else float("nan"),
        )
        return out


class LatticeAccumulator:
    """Arm A's ``lattice_geometry``, accumulated over parameters.

    Every field is an exact sum or count except the ``_from_hist`` quantiles.
    """

    def __init__(self, device) -> None:
        z = lambda n: torch.zeros(n, dtype=torch.float64, device=device)  # noqa: E731
        self.log_edges = torch.tensor(LOG_X_EDGES, dtype=torch.float32, device=device)
        self.lin_edges = torch.tensor(LINEAR_X_EDGES, dtype=torch.float32, device=device)
        self.log_counts = torch.zeros(len(LOG_X_EDGES) + 1, dtype=torch.int64, device=device)
        self.lin_counts = torch.zeros(len(LINEAR_X_EDGES) + 1, dtype=torch.int64, device=device)
        self.step_counts = torch.zeros(N_STEP_BINS + 1, dtype=torch.int64, device=device)
        self.n = 0
        self.n_zero_intent = 0
        self.n_x_below_one = 0
        self.sum_x = z(1)
        self.n_moved = 0
        self.n_multi = 0
        self.sum_steps_moved = z(1)
        self.n_binade_edge = 0
        # Weighted (by ULP^2) and unweighted ratio components.
        self.w_steps_sq = z(1)
        self.w_steps_x_aligned = z(1)
        self.u_steps_sq = z(1)
        self.u_steps_x_aligned = z(1)
        self.w_pred_sq = z(1)
        self.w_pred_x = z(1)
        self.u_pred_sq = z(1)
        self.u_pred_x = z(1)
        self.w_single = z(1)
        self.w_single_x = z(1)
        self.w_multi_steps_sq = z(1)
        self.w_x_above_half = z(1)
        self.w_above_half = z(1)
        # Decay-term diagnostic: is decoupled weight decay visible to the writer?
        self.decay_x_max = 0.0
        self.n_decay_above_half = 0

    def add(self, p_pre: torch.Tensor, intent: torch.Tensor,
            realized: torch.Tensor, decay: torch.Tensor | None = None) -> None:
        # Parameters are 1-D to 3-D; every statistic here is over coordinates,
        # and bincount takes 1-D input only, so flatten up front.
        p_pre = p_pre.reshape(-1)
        intent = intent.reshape(-1)
        realized = realized.reshape(-1)
        if decay is not None:
            decay = decay.reshape(-1)
        ulp = bf16_ulp(p_pre)
        x = torch.abs(intent) / ulp
        signed_steps = torch.round(realized / ulp)
        steps = torch.abs(signed_steps)
        aligned = torch.sign(signed_steps) * torch.sign(intent)
        weight = (ulp.double()) ** 2

        self.n += x.numel()
        positive = x > 0
        self.n_zero_intent += int((~positive).sum())
        self.n_x_below_one += int((x < 1.0).sum())
        self.sum_x += torch.sum(x, dtype=torch.float64)

        log_x = torch.log10(x[positive])
        self.log_counts += torch.bincount(
            torch.bucketize(log_x, self.log_edges, right=True),
            minlength=self.log_counts.numel(),
        )
        inside = x[positive & (x < 1.0)]
        self.lin_counts += torch.bincount(
            torch.bucketize(inside, self.lin_edges, right=True),
            minlength=self.lin_counts.numel(),
        )
        self.step_counts += torch.bincount(
            torch.clamp(steps.long(), max=N_STEP_BINS),
            minlength=N_STEP_BINS + 1,
        )

        moved = steps > 0
        single = steps == 1
        multi = steps >= 2
        above_half = x > 0.5
        self.n_moved += int(moved.sum())
        self.n_multi += int(multi.sum())
        self.sum_steps_moved += torch.sum(steps * moved, dtype=torch.float64)
        # A parameter sitting exactly on a binade edge has |mantissa| == 0.5.
        self.n_binade_edge += int((torch.frexp(p_pre)[0].abs() == 0.5).sum())

        steps_d = steps.double()
        x_d = x.double()
        aligned_d = aligned.double()
        pred = torch.round(x).double()

        self.w_steps_sq += torch.sum(weight * steps_d**2, dtype=torch.float64)
        self.w_steps_x_aligned += torch.sum(
            weight * steps_d * x_d * aligned_d, dtype=torch.float64)
        self.u_steps_sq += torch.sum(steps_d**2, dtype=torch.float64)
        self.u_steps_x_aligned += torch.sum(
            steps_d * x_d * aligned_d, dtype=torch.float64)
        self.w_pred_sq += torch.sum(weight * pred**2, dtype=torch.float64)
        self.w_pred_x += torch.sum(weight * pred * x_d, dtype=torch.float64)
        self.u_pred_sq += torch.sum(pred**2, dtype=torch.float64)
        self.u_pred_x += torch.sum(pred * x_d, dtype=torch.float64)
        self.w_single += torch.sum(weight * single, dtype=torch.float64)
        self.w_single_x += torch.sum(weight * single * x_d, dtype=torch.float64)
        self.w_multi_steps_sq += torch.sum(
            weight * multi * steps_d**2, dtype=torch.float64)
        self.w_x_above_half += torch.sum(
            weight * above_half * x_d, dtype=torch.float64)
        self.w_above_half += torch.sum(weight * above_half, dtype=torch.float64)

        if decay is not None:
            decay_x = torch.abs(decay) / ulp
            self.decay_x_max = max(self.decay_x_max, float(decay_x.max()))
            self.n_decay_above_half += int((decay_x > 0.5).sum())

    @staticmethod
    def _ratio(num: float, den: float) -> float:
        return num / den if den else float("nan")

    def _quantiles_from_hist(self, qs) -> dict:
        counts = self.log_counts.cpu().numpy()[1:-1].astype(np.float64)
        total = counts.sum()
        if total == 0:
            return {str(q): float("nan") for q in qs}
        cum = np.concatenate([[0.0], np.cumsum(counts)]) / total
        edges = LOG_X_EDGES
        out = {}
        for q in qs:
            i = int(np.searchsorted(cum, q, side="left"))
            i = min(max(i, 1), len(edges) - 1)
            lo, hi = cum[i - 1], cum[i]
            frac = (q - lo) / (hi - lo) if hi > lo else 0.0
            out[str(q)] = float(10.0 ** (edges[i - 1] + frac * (edges[i] - edges[i - 1])))
        return out

    def result(self) -> dict:
        f = float
        lin = self.lin_counts.cpu().numpy()[1:-1]
        n_inside = int(lin.sum())
        expected = n_inside / len(lin) if n_inside else 0.0
        chi2 = (f(np.sum((lin - expected) ** 2 / expected)) if expected
                else float("nan"))
        steps_hist = self.step_counts.cpu().numpy()
        q = self._quantiles_from_hist([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        w_steps_sq = f(self.w_steps_sq)
        return {
            "n_coords": self.n,
            "n_zero_intent": self.n_zero_intent,
            "x_below_one_fraction": self.n_x_below_one / self.n if self.n else float("nan"),
            "x_mean": f(self.sum_x) / self.n if self.n else float("nan"),
            "x_quantiles_from_hist": q,
            "x_log10_decade_span_5_95": (
                math.log10(q["0.95"]) - math.log10(q["0.05"])
                if q["0.95"] > 0 and q["0.05"] > 0 else float("nan")
            ),
            "x_log10_hist": {
                "edges": LOG_X_EDGES.tolist(),
                "counts": self.log_counts.cpu().numpy()[1:-1].tolist(),
                "below": int(self.log_counts[0]),
                "above": int(self.log_counts[-1]),
            },
            "x_in_unit_interval_hist": {
                "edges": LINEAR_X_EDGES.tolist(),
                "counts": lin.tolist(),
                "n": n_inside,
                "chi_square_vs_uniform": chi2,
                "degrees_of_freedom": len(lin) - 1,
                "max_bin_over_mean": f(lin.max() / expected) if expected else float("nan"),
                "min_bin_over_mean": f(lin.min() / expected) if expected else float("nan"),
            },
            "step_hist": {
                "edges": STEP_EDGES.tolist(),
                "counts": steps_hist[:N_STEP_BINS].tolist(),
                "above_edges": int(steps_hist[N_STEP_BINS]),
            },
            "zero_step_fraction": (self.n - self.n_moved) / self.n if self.n else float("nan"),
            "abs_step_ge_2_share_of_all": self.n_multi / self.n if self.n else float("nan"),
            "abs_step_ge_2_share_of_moved": (
                self.n_multi / self.n_moved if self.n_moved else float("nan")),
            "mean_abs_step_of_moved": (
                f(self.sum_steps_moved) / self.n_moved if self.n_moved else float("nan")),
            "mean_x_above_half": self._ratio(f(self.w_x_above_half), f(self.w_above_half)),
            "inverse_mean_x_above_half": self._ratio(
                f(self.w_above_half), f(self.w_x_above_half)),
            "single_step_only_ratio": self._ratio(f(self.w_single), f(self.w_single_x)),
            "multi_step_share_of_numerator": self._ratio(
                f(self.w_multi_steps_sq), w_steps_sq),
            "binade_edge_fraction": self.n_binade_edge / self.n if self.n else float("nan"),
            "ratio_from_realized_steps": self._ratio(
                w_steps_sq, f(self.w_steps_x_aligned)),
            "ratio_from_realized_steps_unweighted": self._ratio(
                f(self.u_steps_sq), f(self.u_steps_x_aligned)),
            "ratio_predicted_from_x": self._ratio(f(self.w_pred_sq), f(self.w_pred_x)),
            "ratio_predicted_from_x_unweighted": self._ratio(
                f(self.u_pred_sq), f(self.u_pred_x)),
            "uniform_x_reference": UNIFORM_X_OVERSHOOT,
            "decay_x_max": self.decay_x_max,
            "decay_coords_above_half_ulp": self.n_decay_above_half,
        }
