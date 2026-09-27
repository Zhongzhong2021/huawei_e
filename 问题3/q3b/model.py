"""Local additive evidence model and independently trained controls.

The pair operation adapts the projected elementwise product in MultiBench's
``fusions/common_fusions.py`` (MIT, commit 49f9be224f342de005b7f001281f33df4301eb22).
The registered, bias-free P/Q/V maps and four shared heads here implement the
Plan B specification rather than copying MultiBench's LRTF class.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

MODALITIES = ("T", "A", "V")
PAIRS = (("T", "A"), ("T", "V"), ("A", "V"))


class LocalEncoder(nn.Module):
    def __init__(self, input_dim: int, row_dim: int, hidden_dim: int, window_size: int,
                 encoder_variant: str = "plain"):
        super().__init__()
        if encoder_variant not in ("plain", "residual"):
            raise ValueError(f"unknown encoder_variant: {encoder_variant}")
        self.row = nn.Linear(input_dim, row_dim)
        self.window = nn.Linear(window_size * row_dim, hidden_dim)
        self.encoder_variant = encoder_variant
        if encoder_variant == "residual":
            self.norm = nn.LayerNorm(hidden_dim)
            self.residual = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                          nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        rows = F.gelu(self.row(x)) * observed.unsqueeze(-1)
        hidden = F.gelu(self.window(rows.flatten(start_dim=2)))
        if self.encoder_variant == "residual":
            hidden = hidden + self.residual(self.norm(hidden))
        return hidden


class PairProjection(nn.Module):
    def __init__(self, hidden_dim: int, rank: int):
        super().__init__()
        self.P = nn.Linear(hidden_dim, rank, bias=False)
        self.Q = nn.Linear(hidden_dim, rank, bias=False)
        self.V = nn.Linear(rank, 4, bias=False)
        nn.init.zeros_(self.V.weight)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.V(self.P(left) * self.Q(right))


class _LocalBase(nn.Module):
    def __init__(self, input_dims: dict[str, int], row_dim: int = 32,
                 hidden_dim: int = 64, window_size: int = 3,
                 modalities: tuple[str, ...] = MODALITIES,
                 encoder_variant: str = "plain"):
        super().__init__()
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.modalities = modalities
        self.encoders = nn.ModuleDict({
            m: LocalEncoder(input_dims[m], row_dim, hidden_dim, window_size, encoder_variant)
            for m in modalities
        })

    def _deltas(self, batch: dict) -> dict[str, torch.Tensor]:
        result = {}
        for m in self.modalities:
            x = batch["x"][m]
            observed = batch["observed"][m]
            if x.ndim != 4 or x.shape[2] != self.window_size:
                raise ValueError(f"x[{m}] must have shape [B,K,{self.window_size},D]")
            if observed.shape != x.shape[:3]:
                raise ValueError(f"observed[{m}] shape does not match x[{m}]")
            encoder = self.encoders[m]
            result[m] = encoder(x, observed) - encoder(torch.zeros_like(x), observed)
        return result

    def interaction_parameters(self):
        return iter(())


class PlanB(_LocalBase):
    def __init__(self, input_dims: dict[str, int], row_dim: int = 32,
                 hidden_dim: int = 64, rank: int = 8, window_size: int = 3,
                 encoder_variant: str = "plain"):
        super().__init__(input_dims, row_dim, hidden_dim, window_size,
                         encoder_variant=encoder_variant)
        self.single_heads = nn.ModuleDict({m: nn.Linear(hidden_dim, 4, bias=False)
                                           for m in MODALITIES})
        self.pair_modules = nn.ModuleDict({a + b: PairProjection(hidden_dim, rank)
                                           for a, b in PAIRS})
        self.bias = nn.Parameter(torch.zeros(4))

    def interaction_parameters(self):
        return self.pair_modules.parameters()

    def forward(self, batch: dict, interaction_gain: float = 1.0) -> dict:
        d = self._deltas(batch)
        weight = batch["window_weight"].unsqueeze(-1)
        single = torch.stack([self.single_heads[m](d[m]) * weight for m in MODALITIES], dim=2)
        pair = torch.stack([
            self.pair_modules[a + b](d[a], d[b]) * weight * interaction_gain
            for a, b in PAIRS
        ], dim=2)
        return _additive_output(single, pair, self.bias)


class AdditiveModel(_LocalBase):
    def __init__(self, input_dims: dict[str, int], row_dim: int = 32,
                 hidden_dim: int = 64, window_size: int = 3,
                 modalities: tuple[str, ...] = MODALITIES,
                 encoder_variant: str = "plain"):
        super().__init__(input_dims, row_dim, hidden_dim, window_size, modalities,
                         encoder_variant)
        self.single_heads = nn.ModuleDict({m: nn.Linear(hidden_dim, 4, bias=False)
                                           for m in modalities})
        self.bias = nn.Parameter(torch.zeros(4))
        self.pair_modules = nn.ModuleDict()

    def forward(self, batch: dict, interaction_gain: float = 1.0) -> dict:
        d = self._deltas(batch)
        weight = batch["window_weight"].unsqueeze(-1)
        template = next(iter(d.values()))
        parts = [self.single_heads[m](d[m]) * weight if m in self.modalities
                 else template.new_zeros((*template.shape[:2], 4)) for m in MODALITIES]
        single = torch.stack(parts, dim=2)
        pair = single.new_zeros(single.shape)
        return _additive_output(single, pair, self.bias)


class TextOnlyModel(AdditiveModel):
    def __init__(self, input_dims: dict[str, int], row_dim: int = 32,
                 hidden_dim: int = 64, window_size: int = 3,
                 encoder_variant: str = "plain"):
        super().__init__(input_dims, row_dim, hidden_dim, window_size, ("T",),
                         encoder_variant)


class FreeFusionModel(_LocalBase):
    """Control with no native additive or local explanation contract."""

    def __init__(self, input_dims: dict[str, int], row_dim: int = 32,
                 hidden_dim: int = 64, window_size: int = 3,
                 encoder_variant: str = "plain"):
        super().__init__(input_dims, row_dim, hidden_dim, window_size,
                         encoder_variant=encoder_variant)
        self.fusion = nn.Sequential(nn.Linear(3 * hidden_dim, 32), nn.GELU(), nn.Linear(32, 4))
        self.pair_modules = nn.ModuleDict()

    def forward(self, batch: dict, interaction_gain: float = 1.0) -> dict:
        d = self._deltas(batch)
        weight = batch["window_weight"].unsqueeze(-1)
        pooled = torch.cat([(d[m] * weight).sum(dim=1) for m in MODALITIES], dim=-1)
        score = self.fusion(pooled)
        return {"score": score, "q": score[:, 0],
                "raw_intensity": 3 * torch.tanh(score[:, 0]),
                "aux_logits": score[:, 1:]}


def _additive_output(single: torch.Tensor, pair: torch.Tensor,
                     bias: torch.Tensor) -> dict:
    local = single.clone()
    local[:, :, 0] += 0.5 * (pair[:, :, 0] + pair[:, :, 1])
    local[:, :, 1] += 0.5 * (pair[:, :, 0] + pair[:, :, 2])
    local[:, :, 2] += 0.5 * (pair[:, :, 1] + pair[:, :, 2])
    modal = local.sum(dim=1)
    score = bias + single.sum(dim=(1, 2)) + pair.sum(dim=(1, 2))
    return {"score": score, "q": score[:, 0],
            "raw_intensity": 3 * torch.tanh(score[:, 0]),
            "aux_logits": score[:, 1:], "single": single, "pair": pair,
            "local": local, "modal": modal, "bias": bias}


def build_model(config: dict) -> nn.Module:
    """Build from the full run config or its ``model`` section."""
    c = config.get("model", config)
    args = {key: c[key] for key in ("input_dims", "row_dim", "hidden_dim", "window_size",
                                   "encoder_variant")
            if key in c}
    kind = c.get("kind", "plan_b")
    if kind == "bert_fusion":
        from .bert_fusion import BertFusionModel
        args.pop("encoder_variant", None)
        return BertFusionModel(**args, **{key: c[key] for key in
            ("dropout", "trainable_layers", "text_only", "pretrained_name") if key in c})
    if kind in ("contextual_evidence", "sequence_fusion", "bert_evidence"):
        extra = {key: c[key] for key in ("dropout", "pooling", "fusion", "trainable_layers",
                                        "text_only", "pretrained_name") if key in c}
        if kind == "contextual_evidence":
            from .contextual_model import ContextualEvidenceModel
            return ContextualEvidenceModel(**args, rank=c.get("rank", 8), **extra)
        if kind == "sequence_fusion":
            from .sequence_models import SequenceFusionModel
            return SequenceFusionModel(**args, rank=c.get("rank", 8), **extra)
        from .bert_model import BertEvidenceModel
        args.pop("encoder_variant", None)
        return BertEvidenceModel(**args, rank=c.get("rank", 8), **extra)
    if kind == "plan_b":
        return PlanB(**args, rank=c.get("rank", 8))
    if kind == "additive":
        return AdditiveModel(**args)
    if kind == "text_only":
        return TextOnlyModel(**args)
    if kind == "free_fusion":
        return FreeFusionModel(**args)
    raise ValueError(f"unknown model kind: {kind}")
