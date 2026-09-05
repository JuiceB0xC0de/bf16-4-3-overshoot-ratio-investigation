"""Vendored Muon reference implementation.

Source: https://github.com/KellerJordan/Muon, ``muon.py`` on ``master``,
retrieved 2026-09-05.  ``zeropower_via_newtonschulz5`` and ``muon_update`` are
copied verbatim from that file; only ``SingleDeviceMuonWithIntent`` is new, and
it differs from the upstream ``SingleDeviceMuon`` in exactly one respect: it
keeps a reference to the update tensor it just applied, so the observer can
read back the direction the optimizer actually used.  The arithmetic, the
dtype of the Newton-Schulz iteration (bfloat16, as upstream), the momentum
convention, the RMS scaling, and the two-write ``mul_`` then ``add_`` order are
untouched.

Vendored rather than pip-installed so the job has no install step and the exact
code is in the experiment's own history.
"""

from __future__ import annotations

import torch

# ---------------------------------------------------------------- upstream --


def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:  # for the case of conv filters
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
    return update


# ------------------------------------------------------------------- local --


class SingleDeviceMuonWithIntent(torch.optim.Optimizer):
    """Upstream ``SingleDeviceMuon`` plus a readable record of the update.

    Upstream discards ``update`` after applying it.  The overshoot measurement
    needs the direction the optimizer *intended*, in the dtype it computed it
    in, so it can be upcast to fp32 and compared against what the bf16 write
    actually stored.  ``state["last_update"]`` holds exactly the tensor that
    was passed to ``p.add_``; nothing else changes.
    """

    def __init__(self, params, lr=0.02, weight_decay=0.0, momentum=0.95,
                 ns_steps=5, nesterov=True):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        ns_steps=ns_steps, nesterov=nesterov)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    raise RuntimeError(
                        "Muon received a parameter with no gradient; the "
                        "roster claims every parameter is updated every step"
                    )
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                update = muon_update(
                    p.grad, state["momentum_buffer"], beta=group["momentum"],
                    ns_steps=group["ns_steps"], nesterov=group["nesterov"],
                )
                update = update.reshape(p.shape)
                # Same two writes, same order, as upstream and as AdamW:
                # decoupled decay multiply, then one additive update write.
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"])
                state["last_update"] = update
        return loss
