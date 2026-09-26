"""Learned ordinal thresholds and uncertainty-weighted native evidence.

This is a competition adaptation, not a reproduction of CORAL or TMSON.
Variance is supervised for unimodal intensity predictions; normalized precision
weights the native score terms. It is not a calibrated fused output variance,
and neither precision weights nor coefficients are causal contributions.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .model import MODALITIES, PAIRS, _additive_output
from .native_next import NativeNextModel
from .native_selfmm import NativeSelfMM


class OrdinalUncertaintyModel(NativeNextModel):
    def __init__(self, *, ordinal_head=False, uncertainty_fusion=False, **kwargs):
        super().__init__(**kwargs)
        self.ordinal_head = bool(ordinal_head)
        self.uncertainty_fusion = bool(uncertainty_fusion)
        if self.ordinal_head:
            self.ordinal_center = nn.Parameter(torch.tensor(0.0))
            self.ordinal_gap_raw = nn.Parameter(torch.tensor(-0.310264))
        if self.uncertainty_fusion:
            self.variance_heads = nn.ModuleList([nn.Linear(8, 1) for _ in MODALITIES])
            for head in self.variance_heads:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
            self.native_explanation_scope = "dynamic_precision_weighted_contextual_readout"
        if self.ordinal_head:
            self.classification_explanation_scope = "ordinal_shared_score_and_thresholds"

    def ordinal_distribution(self, score):
        gap = F.softplus(self.ordinal_gap_raw) + .05
        thresholds = torch.stack((self.ordinal_center - gap / 2,
                                  self.ordinal_center + gap / 2))
        cumulative_logits = score[:, None] - thresholds
        a, b = cumulative_logits.unbind(-1)
        # Stable log(sigmoid(a)-sigmoid(b)); gap > 0 enforces ordering.
        neutral = F.logsigmoid(a) + F.logsigmoid(-b) + torch.log(-torch.expm1(-gap))
        log_probs = torch.stack((F.logsigmoid(-a), neutral, F.logsigmoid(b)), -1)
        return log_probs, cumulative_logits, thresholds

    def forward(self, batch, interaction_gain=1.0):
        out = super().forward(batch, interaction_gain)
        if self.uncertainty_fusion:
            uni = out["unimodal_scores"]
            # Features use magnitudes as well as signed sums; no validation labels.
            activity = out["single"].abs().sum(1)
            features = torch.cat((uni, activity), -1)
            log_var = torch.stack([head(features[:, i]).squeeze(-1)
                                   for i, head in enumerate(self.variance_heads)], -1)
            log_var = -1 + 4 * torch.tanh(log_var)  # finite variance in [exp(-5),exp(3)]
            active = out["modality_keep"]
            precision = torch.exp(-log_var) * active
            weights = precision / precision.sum(-1, keepdim=True).clamp_min(1e-12)
            # Equal reliability reproduces the original additive scores exactly.
            scales = weights * active.sum(-1, keepdim=True)
            pair_scales = torch.stack([
                (scales[:, MODALITIES.index(a)] * scales[:, MODALITIES.index(b)]
                 ).clamp_min(1e-20).sqrt()
                * active[:, MODALITIES.index(a)] * active[:, MODALITIES.index(b)]
                for a, b in PAIRS], -1)
            weighted = _additive_output(out["single"] * scales[:, None, :, None],
                                        out["pair"] * pair_scales[:, None, :, None],
                                        self.bias)
            out.update(weighted)
            out.update(unimodal_log_variance=log_var,
                       unimodal_intensity=3 * torch.tanh(uni[..., 0]),
                       precision_weights=weights,
                       variance=torch.exp(log_var))
        if self.ordinal_head:
            shared_score = out["score"][:, 3] - out["score"][:, 1]
            log_probs, cumulative, thresholds = self.ordinal_distribution(shared_score)
            out.update(aux_logits=log_probs, ordinal_score=shared_score,
                       ordinal_cumulative_logits=cumulative,
                       ordinal_thresholds=thresholds)
        return out

    def training_loss(self, out, batch, epoch):
        loss = super().training_loss(out, batch, epoch)
        if self.ordinal_head:
            weight = float(self.train_config.get("classification_weight", .5))
            labels = batch["class_label"]
            targets = torch.stack((labels > 0, labels > 1), -1).to(out["aux_logits"].dtype)
            # Replace the fused three-class CE with the two cumulative BCE tasks.
            loss = loss - weight * F.cross_entropy(out["aux_logits"], labels)
            loss = loss + weight * F.binary_cross_entropy_with_logits(
                out["ordinal_cumulative_logits"], targets)
        if self.uncertainty_fusion:
            residual = out["unimodal_intensity"] - batch["y"][:, None]
            log_var = out["unimodal_log_variance"]
            nll = .5 * (torch.exp(-log_var) * residual.square() + log_var)
            observed = out["modality_available"]
            loss = loss + float(self.train_config.get("uncertainty_weight", .1)) * (
                (nll * observed).sum() / observed.sum().clamp_min(1))
        return loss


class HybridOrdinalUncertaintyModel(OrdinalUncertaintyModel, NativeSelfMM):
    """The same ordinal/precision readout on the source-aligned temporal model."""


def build(config):
    model = config["model"]
    keys = ("input_dims", "row_dim", "hidden_dim", "rank", "window_size", "dropout",
            "trainable_layers", "pooling", "text_only", "pretrained_name",
            "modality_dropout", "ordinal_head", "uncertainty_fusion",
            "temporal_dim", "pseudo_weight")
    factory = HybridOrdinalUncertaintyModel if model.get("transfer_backbone") else OrdinalUncertaintyModel
    return factory(**{k: model[k] for k in keys if k in model},
                   train_config=config.get("train", {}))
