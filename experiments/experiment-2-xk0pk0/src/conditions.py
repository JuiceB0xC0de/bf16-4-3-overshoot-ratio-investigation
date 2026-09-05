"""Model loading, parameter roster, and the three optimizer conditions.

The variable under test is the update distribution an optimizer generates.
Everything downstream of that -- the bf16 write path -- is held fixed by
construction: all three conditions apply exactly two in-place writes per
parameter per step, in the same order,

    p.mul_(1 - lr*wd)        # decoupled decay
    p.<additive update>      # AdamW addcdiv_, SGD add_, Muon add_

which is what ``torch.optim.AdamW`` does natively and what upstream Muon does
natively.  ``torch.optim.SGD`` applies weight decay as coupled L2 into the
gradient instead, which at these magnitudes would *dominate* SGD's update
(wd*|p| ~ 2e-4 against a clipped per-coordinate gradient RMS ~7e-5), so SGD is
constructed with ``weight_decay=0`` and given the same decoupled decay multiply
as the other two.  That keeps weight decay a shared constant rather than a
confound, and the observer records ``decay_x_max`` so the claim that decay is
invisible to the bf16 writer is measured rather than assumed.

Intent is defined uniformly across optimizers as the fp32 evaluation of the
optimizer's own update rule applied to its own *native post-step state*,
upcast to fp32.  For AdamW that is exactly the published definition in
``analyze_training_optimizer_delivery_observer.py::_ideal_components``, so the
baseline bridges to the recorded numbers; for SGD and Muon it is the direct
analogue.  Defining intent this way matters: the native optimizers keep their
state in the parameter dtype (bfloat16 here) and Muon runs Newton-Schulz in
bfloat16, so recomputing a "clean" fp32 direction would fold the optimizer's
own low-precision arithmetic into the residual and misattribute it to the
writer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import torch

MODEL_ID = "state-spaces/mamba3-siso-187m"
MODEL_REVISION = "6792c27c00f3bb41506db1066dcd1c51bb0f4b02"

# The frozen control's settings.
ADAMW_LR = 2e-4
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.999)
EPS = 1e-8
CLIP_MAX_NORM = 1.0
SEED = 1234

# Provisional learning rates for the scale-matching probe.  SGD and Muon
# updates are linear in lr, so the probe only has to be in the right ballpark.
SGD_PROBE_LR = 0.05
MUON_PROBE_LR = 0.02
SGD_MOMENTUM = 0.9
MUON_MOMENTUM = 0.95

# Structure of the checkpoint, asserted at load so a silently different model
# cannot masquerade as the recorded one.  From the published per-tensor table.
EXPECTED_N_PARAMS = 146
EXPECTED_NUMEL = 186_849_600
EXPECTED_FP32_PARAMS = 24
EXPECTED_FP32_NUMEL = 576
EXPECTED_EMBEDDING_NUMEL = 98_500_608
FP32_SUFFIXES = (".mixer.D", ".mixer.dt_bias")
EMBEDDING_NAME = "backbone.embedding.weight"


def set_seed(seed: int = SEED) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(device: str = "cuda", dtype=torch.bfloat16):
    """Build the model explicitly from the pinned revision's own files.

    ``MambaLMHeadModel.from_pretrained`` does not expose a revision argument, so
    the config and weights are fetched at the pinned commit and the model is
    constructed directly.  That makes the revision a hard fact rather than
    whatever the hub currently serves for the bare model id.
    """
    from huggingface_hub import hf_hub_download
    from mamba_ssm.models.config_mamba import MambaConfig
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

    config_path = hf_hub_download(MODEL_ID, "config.json", revision=MODEL_REVISION)
    weights_path = hf_hub_download(MODEL_ID, "pytorch_model.bin", revision=MODEL_REVISION)
    with open(config_path) as fh:
        raw_config = json.load(fh)
    config = MambaConfig(**raw_config)
    model = MambaLMHeadModel(config, device=device, dtype=dtype)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Tied embeddings mean lm_head.weight is not a separate stored tensor.
    unexpected = [k for k in unexpected]
    missing = [k for k in missing if not k.startswith("lm_head.")]
    if missing or unexpected:
        raise RuntimeError(f"state dict mismatch: missing={missing} unexpected={unexpected}")
    model.train()
    return model, raw_config


@dataclass
class ParamRecord:
    index: int
    name: str
    param: torch.nn.Parameter
    numel: int
    ndim: int
    native_fp32: bool
    muon_covered: bool
    updater: str = "adamw"


def build_roster(model) -> list:
    """One record per distinct optimizer parameter, in ``named_parameters`` order.

    Muon coverage follows the upstream convention: two-dimensional hidden
    weights only, with the input embedding (tied to the output head here)
    excluded and left to AdamW.  This set is defined once and applied
    identically in every condition, so the restricted comparison is over the
    same coordinates whatever optimizer produced the update.
    """
    records = []
    for i, (name, p) in enumerate(model.named_parameters()):
        native_fp32 = p.dtype == torch.float32
        if native_fp32 != name.endswith(FP32_SUFFIXES):
            raise RuntimeError(
                f"{name}: dtype {p.dtype} disagrees with the recorded fp32 "
                f"suffix convention {FP32_SUFFIXES}"
            )
        covered = p.ndim == 2 and name != EMBEDDING_NAME and "lm_head" not in name
        records.append(ParamRecord(i, name, p, p.numel(), p.ndim, native_fp32, covered))

    total = sum(r.numel for r in records)
    n_fp32 = sum(1 for r in records if r.native_fp32)
    fp32_numel = sum(r.numel for r in records if r.native_fp32)
    embedding = [r for r in records if r.name == EMBEDDING_NAME]
    problems = []
    if len(records) != EXPECTED_N_PARAMS:
        problems.append(f"n_params {len(records)} != {EXPECTED_N_PARAMS}")
    if total != EXPECTED_NUMEL:
        problems.append(f"numel {total} != {EXPECTED_NUMEL}")
    if n_fp32 != EXPECTED_FP32_PARAMS:
        problems.append(f"fp32 params {n_fp32} != {EXPECTED_FP32_PARAMS}")
    if fp32_numel != EXPECTED_FP32_NUMEL:
        problems.append(f"fp32 numel {fp32_numel} != {EXPECTED_FP32_NUMEL}")
    if not embedding or embedding[0].numel != EXPECTED_EMBEDDING_NUMEL:
        problems.append(f"embedding numel != {EXPECTED_EMBEDDING_NUMEL}")
    if problems:
        raise RuntimeError("loaded model does not match the recorded structure: "
                           + "; ".join(problems))
    return records


def roster_summary(records: list) -> dict:
    covered = [r for r in records if r.muon_covered]
    return {
        "n_params": len(records),
        "numel": sum(r.numel for r in records),
        "n_bf16_params": sum(1 for r in records if not r.native_fp32),
        "bf16_numel": sum(r.numel for r in records if not r.native_fp32),
        "n_fp32_params": sum(1 for r in records if r.native_fp32),
        "fp32_numel": sum(r.numel for r in records if r.native_fp32),
        "n_muon_covered_params": len(covered),
        "muon_covered_numel": sum(r.numel for r in covered),
        "muon_covered_share_of_numel": (
            sum(r.numel for r in covered) / sum(r.numel for r in records)),
        "muon_covered_names_sample": [r.name for r in covered[:6]],
        "muon_excluded_2d": [r.name for r in records
                             if r.ndim == 2 and not r.muon_covered],
        "shapes_by_ndim": {
            str(d): sum(1 for r in records if r.ndim == d)
            for d in sorted({r.ndim for r in records})
        },
    }


class Condition:
    """One optimizer condition: the real step, plus its own intent rule."""

    def __init__(self, name: str, records: list, lr: float,
                 weight_decay: float = WEIGHT_DECAY):
        self.name = name
        self.records = records
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizers = []
        self._decay_by_hand = []       # params whose decay this class applies
        self._muon = None
        self._adamw = None

        if name == "adamw":
            for r in records:
                r.updater = "adamw"
            self._adamw = torch.optim.AdamW(
                [r.param for r in records], lr=lr, betas=BETAS, eps=EPS,
                weight_decay=weight_decay)
            self.optimizers = [self._adamw]
        elif name == "sgd":
            for r in records:
                r.updater = "sgd"
            self._sgd = torch.optim.SGD(
                [r.param for r in records], lr=lr, momentum=SGD_MOMENTUM,
                dampening=0.0, weight_decay=0.0, nesterov=False)
            self._decay_by_hand = [r.param for r in records]
            self.optimizers = [self._sgd]
        elif name == "muon":
            from vendor_muon import SingleDeviceMuonWithIntent

            covered = [r for r in records if r.muon_covered]
            rest = [r for r in records if not r.muon_covered]
            for r in covered:
                r.updater = "muon"
            for r in rest:
                r.updater = "adamw"
            self._muon = SingleDeviceMuonWithIntent(
                [r.param for r in covered], lr=lr, weight_decay=weight_decay,
                momentum=MUON_MOMENTUM)
            # Non-matrix parameters keep the frozen control's AdamW settings:
            # the standard convention, and it leaves the fallback parameters
            # treated exactly as in the AdamW condition.
            self._adamw = torch.optim.AdamW(
                [r.param for r in rest], lr=ADAMW_LR, betas=BETAS, eps=EPS,
                weight_decay=weight_decay)
            self.optimizers = [self._muon, self._adamw]
        else:
            raise ValueError(f"unknown condition {name!r}")

    def config(self) -> dict:
        out = {"condition": self.name, "lr": self.lr,
               "weight_decay": self.weight_decay,
               "decay_semantics": "decoupled (all conditions)"}
        if self.name == "sgd":
            out.update(momentum=SGD_MOMENTUM, nesterov=False, dampening=0.0)
        if self.name == "muon":
            out.update(momentum=MUON_MOMENTUM, ns_steps=5, nesterov=True,
                       fallback="adamw", fallback_lr=ADAMW_LR)
        if self.name == "adamw":
            out.update(betas=list(BETAS), eps=EPS)
        return out

    @torch.no_grad()
    def step(self) -> None:
        # Decoupled decay for the optimizer that does not apply it itself.
        # AdamW and Muon each do their own mul_ inside step().
        for p in self._decay_by_hand:
            p.mul_(1 - self.lr * self.weight_decay)
        for opt in self.optimizers:
            opt.step()

    def zero_grad(self) -> None:
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)

    def _group_for(self, opt, param):
        for group in opt.param_groups:
            for p in group["params"]:
                if p is param:
                    return group
        raise KeyError("parameter not in optimizer")

    @torch.no_grad()
    def intent(self, record: ParamRecord, p_pre32: torch.Tensor):
        """fp32 intended update and its decay component, from post-step state.

        Returns ``(intent, decay)`` where ``intent = decay + direction`` and
        ``decay = -(lr*wd)*p_pre`` for every condition.
        """
        p = record.param
        if record.updater == "adamw":
            opt = self._adamw
            group = self._group_for(opt, p)
            state = opt.state[p]
            lr = float(group["lr"])
            wd = float(group["weight_decay"])
            eps = float(group["eps"])
            beta1, beta2 = group["betas"]
            t = float(state["step"])
            if t <= 0:
                raise RuntimeError(f"{record.name}: AdamW state step is {t}")
            bc1 = 1.0 - beta1 ** t
            bc2 = 1.0 - beta2 ** t
            m = state["exp_avg"].to(torch.float32)
            v = state["exp_avg_sq"].to(torch.float32)
            decay = p_pre32 * -(lr * wd)
            denom = v.sqrt() / float(bc2 ** 0.5) + eps
            adaptive = m / denom * -(lr / bc1)
            return decay + adaptive, decay

        if record.updater == "sgd":
            state = self._sgd.state[p]
            buf = state["momentum_buffer"].to(torch.float32)
            decay = p_pre32 * -(self.lr * self.weight_decay)
            return decay + buf * -self.lr, decay

        if record.updater == "muon":
            state = self._muon.state[p]
            update = state["last_update"].to(torch.float32)
            group = self._group_for(self._muon, p)
            lr = float(group["lr"])
            wd = float(group["weight_decay"])
            decay = p_pre32 * -(lr * wd)
            return decay + update * -lr, decay

        raise ValueError(f"unknown updater {record.updater!r}")
