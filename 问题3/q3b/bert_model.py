"""Pretrained BERT with native local evidence and pairwise readout.

Uses Hugging Face Transformers' Apache-2.0 BERT implementation and the
``bert-base-uncased`` checkpoint (Apache-2.0). Token input organization also
matches MMSA's MIT-licensed BertTextEncoder; no MMSA source is copied here.
Local text values are contextual readouts, not independent word effects.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from .bert_init import load_bert

from .model import LocalEncoder, MODALITIES, PAIRS, PairProjection, _additive_output


class BertEvidenceModel(nn.Module):
    local_deletion_is_exact = False
    native_explanation_scope = "contextual_readout"

    def __init__(self, input_dims: dict[str, int], row_dim: int = 64,
                 hidden_dim: int = 64, rank: int = 8, window_size: int = 1,
                 dropout: float = .2, trainable_layers: int = 2,
                 pooling: str = "mean", text_only: bool = False,
                 pretrained_name: str = "bert-base-uncased"):
        super().__init__()
        if pooling not in ("mean", "attention"):
            raise ValueError("pooling must be mean or attention")
        if not 0 <= dropout < 1 or window_size < 1:
            raise ValueError("invalid dropout or window_size")
        self.bert = load_bert(pretrained_name)
        layers = self.bert.encoder.layer
        if not 0 <= trainable_layers <= len(layers):
            raise ValueError("trainable_layers exceeds the BERT encoder depth")
        self.trainable_layers = trainable_layers
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.pooling = pooling
        self.text_only = text_only
        self.modalities = ("T",) if text_only else MODALITIES
        for parameter in self.bert.parameters():
            parameter.requires_grad_(False)
        for layer in list(layers)[len(layers) - trainable_layers:]:
            for parameter in layer.parameters():
                parameter.requires_grad_(True)
        dims = {**input_dims, "T": self.bert.config.hidden_size}
        self.encoders = nn.ModuleDict({
            m: LocalEncoder(dims[m], row_dim, hidden_dim, window_size)
            for m in self.modalities
        })
        self.dropout = nn.Dropout(dropout)
        self.single_heads = nn.ModuleDict({
            m: nn.Linear(hidden_dim, 4, bias=False) for m in self.modalities
        })
        self.attention_heads = nn.ModuleDict({
            m: nn.Linear(hidden_dim, 1, bias=False) for m in self.modalities
        }) if pooling == "attention" else nn.ModuleDict()
        self.pair_modules = nn.ModuleDict({
            a + b: PairProjection(hidden_dim, rank) for a, b in PAIRS
        }) if not text_only else nn.ModuleDict()
        self.bias = nn.Parameter(torch.zeros(4))
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, "bert"):
            # Freezing parameters alone does not disable dropout. Keep the
            # frozen prefix deterministic while tuning the selected suffix.
            self.bert.embeddings.eval()
            layers = self.bert.encoder.layer
            for layer in list(layers)[:len(layers) - self.trainable_layers]:
                layer.eval()
            pooler = getattr(self.bert, "pooler", None)
            if pooler is not None:
                pooler.eval()
        return self

    def interaction_parameters(self):
        return self.pair_modules.parameters()

    def _text_rows(self, batch: dict) -> torch.Tensor:
        bert = batch["bert_inputs"]
        source = batch["source_index"]
        content = batch["content"]
        if bert.ndim != 3 or bert.shape[1] != 3:
            raise ValueError("bert_inputs must have shape [B,3,L]")
        if source.shape != content.shape or source.shape[0] != bert.shape[0]:
            raise ValueError("source_index/content shape mismatch")
        if source.shape[2] != self.window_size:
            raise ValueError("source_index has wrong window_size")
        if torch.any(content & ((source < 0) | (source >= bert.shape[2]))):
            raise ValueError("content source_index is outside BERT sequence")
        hidden = self.bert(input_ids=bert[:, 0].long(),
                           attention_mask=bert[:, 1].long(),
                           token_type_ids=bert[:, 2].long()).last_hidden_state
        index = source.clamp(0, bert.shape[2] - 1).flatten(1)
        rows = hidden.gather(1, index.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]))
        rows = rows.reshape(*source.shape, hidden.shape[-1])
        rows = F.layer_norm(rows, (rows.shape[-1],))
        return rows * content.unsqueeze(-1)

    def _weights(self, hidden: torch.Tensor, observed: torch.Tensor, modality: str):
        counts = observed.sum(dim=-1).to(hidden.dtype)
        valid = counts > 0
        if self.pooling == "mean":
            weights = counts / counts.sum(dim=1, keepdim=True).clamp_min(1)
        else:
            logits = self.attention_heads[modality](hidden).squeeze(-1)
            weights = logits.masked_fill(~valid, -1e4).softmax(dim=1) * valid
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return weights, valid

    def forward(self, batch: dict, interaction_gain: float = 1.0) -> dict:
        text = self._text_rows(batch)
        d, weight, valid = {}, {}, {}
        for m in self.modalities:
            x = text if m == "T" else batch["x"][m]
            observed = batch["observed"][m]
            if x.shape[:3] != observed.shape:
                raise ValueError(f"{m} observations do not match feature rows")
            encoder = self.encoders[m]
            if m == "T" and "bert_readout_mask" in batch:
                readout = batch["bert_readout_mask"]
                if readout.shape != observed.shape:
                    raise ValueError("bert_readout_mask has wrong shape")
                # This masks contextual readout positions without pretending
                # that the source words were removed and BERT recomputed.
                x = x * readout.unsqueeze(-1)
            delta = encoder(x, observed) - encoder(torch.zeros_like(x), observed)
            d[m] = self.dropout(delta) * observed.any(dim=-1, keepdim=True)
            weight[m], valid[m] = self._weights(d[m], observed, m)
        template = d["T"]
        zero = template.new_zeros((*template.shape[:2], 4))
        single = torch.stack([
            self.single_heads[m](d[m]) * weight[m].unsqueeze(-1)
            if m in self.modalities else zero for m in MODALITIES
        ], dim=2)
        pairs = []
        for a, b in PAIRS:
            if self.text_only:
                pairs.append(zero)
                continue
            joint_valid = valid[a] & valid[b]
            joint = (weight[a] * weight[b]).clamp_min(1e-20).sqrt() * joint_valid
            joint = joint / joint.sum(dim=1, keepdim=True).clamp_min(1e-12)
            pairs.append(self.pair_modules[a + b](d[a], d[b])
                         * joint.unsqueeze(-1) * interaction_gain)
        return _additive_output(single, torch.stack(pairs, dim=2), self.bias)
