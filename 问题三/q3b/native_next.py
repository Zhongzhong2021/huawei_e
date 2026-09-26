"""Training extensions for the native BERT evidence readout.

The evaluation structure is unchanged: four scores equal bias plus local
single-modality and explicit pair terms. Additional supervision reuses those
scores; structured modality dropout operates only during training.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from .bert_model import BertEvidenceModel
from .model import MODALITIES, PAIRS, _additive_output
from .train import _ordinal_loss


class NativeNextModel(BertEvidenceModel):
    def __init__(self, *, train_config: dict | None = None,
                 modality_dropout: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        if not 0 <= modality_dropout < 1:
            raise ValueError("modality_dropout must lie in [0,1)")
        self.modality_dropout = float(modality_dropout)
        self.train_config = dict(train_config or {})
        for name in ("unimodal_weight", "neutral_weight", "polarity_weight"):
            if self.train_config.get(name, 0) < 0:
                raise ValueError(f"{name} cannot be negative")

    def _modality_keep(self, available: torch.Tensor):
        if not self.training or not self.modality_dropout:
            return available
        draw = torch.rand(available.shape, device=available.device)
        keep = (draw >= self.modality_dropout) & available
        # Keep one actually observed modality when all observed ones were
        # sampled out; a fully unobserved sample remains fully unobserved.
        restore = available.any(dim=1) & ~keep.any(dim=1)
        choice = draw.masked_fill(~available, -1).argmax(dim=1)
        fallback = F.one_hot(choice, num_classes=3).bool() & restore.unsqueeze(-1)
        return keep | fallback

    def forward(self, batch: dict, interaction_gain: float = 1.0) -> dict:
        full = super().forward(batch, interaction_gain=interaction_gain)
        available = torch.stack([
            batch["observed"][m].any(dim=(1, 2)) if m in self.modalities
            else torch.zeros_like(batch["observed"][m].any(dim=(1, 2)))
            for m in MODALITIES
        ], dim=1)
        unimodal = full["single"].sum(dim=1) + self.bias
        keep = self._modality_keep(available)
        if self.training and self.modality_dropout:
            single = full["single"] * keep[:, None, :, None]
            pair_keep = torch.stack([keep[:, MODALITIES.index(a)] & keep[:, MODALITIES.index(b)]
                                     for a, b in PAIRS], dim=1)
            pair = full["pair"] * pair_keep[:, None, :, None]
            out = _additive_output(single, pair, self.bias)
        else:
            out = full
        out.update(unimodal_scores=unimodal, modality_available=available,
                   modality_keep=keep)
        return out

    def training_loss(self, out: dict, batch: dict, epoch: int):
        del epoch
        cfg = self.train_config
        delta = float(cfg.get("huber_delta", 1.0))
        ce_weight = float(cfg.get("classification_weight", .5))
        regression = F.huber_loss(out["raw_intensity"], batch["y"], delta=delta)
        ce = F.cross_entropy(out["aux_logits"], batch["class_label"])
        loss = regression + ce_weight * ce
        ordinal_weight = float(cfg.get("ordinal_weight", 0.0))
        if ordinal_weight:
            loss = loss + ordinal_weight * _ordinal_loss(
                out["raw_intensity"], batch["class_label"],
                float(cfg.get("ordinal_band", .3)), float(cfg.get("ordinal_temperature", .2)))
        loss = loss + float(cfg.get("interaction_l1", .001)) * out["pair"][..., 0].abs().sum((1, 2)).mean()
        unimodal_weight = float(cfg.get("unimodal_weight", 0.0))
        if unimodal_weight:
            scores = out["unimodal_scores"]
            target = batch["y"].unsqueeze(-1).expand_as(scores[..., 0])
            uni_reg = F.huber_loss(3 * torch.tanh(scores[..., 0]), target,
                                 reduction="none", delta=delta)
            labels = batch["class_label"].unsqueeze(-1).expand_as(target)
            uni_ce = F.cross_entropy(scores[..., 1:].reshape(-1, 3), labels.reshape(-1),
                                     reduction="none").reshape_as(target)
            observed = out["modality_available"].to(scores.dtype)
            uni_loss = ((uni_reg + ce_weight * uni_ce) * observed).sum() / observed.sum().clamp_min(1)
            loss = loss + unimodal_weight * uni_loss
        logits, labels = out["aux_logits"], batch["class_label"]
        neutral_weight = float(cfg.get("neutral_weight", 0.0))
        if neutral_weight:
            neutral_log_odds = logits[:, 1] - torch.logsumexp(logits[:, [0, 2]], dim=-1)
            loss = loss + neutral_weight * F.binary_cross_entropy_with_logits(
                neutral_log_odds, (labels == 1).to(logits.dtype))
        polarity_weight = float(cfg.get("polarity_weight", 0.0))
        if polarity_weight:
            nonneutral = labels != 1
            polarity = F.binary_cross_entropy_with_logits(
                logits[:, 2] - logits[:, 0], (labels == 2).to(logits.dtype), reduction="none")
            loss = loss + polarity_weight * (polarity * nonneutral).sum() / nonneutral.sum().clamp_min(1)
        return loss


def build(config: dict):
    model = config["model"]
    keys = ("input_dims", "row_dim", "hidden_dim", "rank", "window_size", "dropout",
            "trainable_layers", "pooling", "text_only", "pretrained_name", "modality_dropout")
    return NativeNextModel(**{key: model[key] for key in keys if key in model},
                           train_config=config.get("train", {}))
