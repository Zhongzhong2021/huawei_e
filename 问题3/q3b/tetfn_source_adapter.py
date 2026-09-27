"""Direct MMSA TETFN wrapper (MIT, commit a94e65d07fa1ae0d44e552390074b29b0898edfd).

This package includes the minimal MMSA TETFN subtree. Its BERT encoder uses
local config or supplied pretrained weights; the model's numerical structure
is unchanged. The loader avoids broad MMSA initializers and their unrelated
optional dependencies.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch
from torch import nn


def _source_class():
    root = Path(__file__).resolve().parents[1] / "third_party/MMSA/src/MMSA"
    prefix = "_q3b_mmsa_tetfn"
    for suffix, path in (("", root), (".models", root / "models"),
                         (".models.multiTask", root / "models/multiTask"),
                         (".models.subNets", root / "models/subNets"),
                         (".models.subNets.transformers_encoder", root / "models/subNets/transformers_encoder")):
        name = prefix + suffix
        if name not in sys.modules:
            package = ModuleType(name)
            package.__path__ = [str(path)]
            sys.modules[name] = package
    subnets = sys.modules[prefix + ".models.subNets"]
    subnets.BertTextEncoder = importlib.import_module(
        prefix + ".models.subNets.BertTextEncoder").BertTextEncoder
    return importlib.import_module(prefix + ".models.multiTask.TETFN").TETFN


class TETFNSourceAdapter(nn.Module):
    native_explanation_scope = "none"
    local_deletion_is_exact = False

    def __init__(self, input_dims, dst_width=50, heads=5, dropout=.1,
                 post_fusion_dim=64, pretrained_name="bert-base-uncased"):
        super().__init__()
        if dst_width < 1 or heads < 1 or dst_width % heads or not 0 <= dropout < 1:
            raise ValueError("invalid width, heads, or dropout")
        args = SimpleNamespace(
            need_data_aligned=True, use_finetune=True, transformers="bert",
            pretrained=pretrained_name, feature_dims=(input_dims["T"], input_dims["A"], input_dims["V"]),
            a_lstm_hidden_size=32, a_lstm_layers=1, a_lstm_dropout=0.0,
            v_lstm_hidden_size=32, v_lstm_layers=1, v_lstm_dropout=0.0,
            conv1d_kernel_size_l=1, conv1d_kernel_size_a=1,
            dst_feature_dims=dst_width, nheads=heads,
            attn_dropout=dropout, attn_dropout_a=0.0, attn_dropout_v=dropout,
            relu_dropout=dropout, res_dropout=0.0, embed_dropout=dropout,
            post_fusion_dropout=dropout, post_fusion_dim=post_fusion_dim,
            post_text_dropout=dropout, post_text_dim=64,
            post_audio_dropout=0.0, post_audio_dim=32,
            post_video_dropout=dropout, post_video_dim=16)
        self.source = _source_class()(args)
        self.classifier = nn.Linear(post_fusion_dim, 3)

    @staticmethod
    def _restore(x, observed, source, length):
        if x.shape[:3] != source.shape or observed.shape != source.shape:
            raise ValueError("feature, observed, and source_index shapes disagree")
        index = source.flatten(1)
        valid = observed.flatten(1)
        if torch.any(valid & ((index < 0) | (index >= length))):
            raise ValueError("observed source_index outside BERT sequence")
        dense = x.new_zeros(x.shape[0], length, x.shape[-1])
        dense.scatter_add_(1, index.clamp(0, length - 1).unsqueeze(-1).expand(-1, -1, x.shape[-1]),
                           x.flatten(1, 2) * valid.unsqueeze(-1))
        return dense

    def forward(self, batch, interaction_gain=1.0):
        bert, source = batch["bert_inputs"], batch["source_index"]
        if bert.ndim != 3 or bert.shape[1] != 3 or source.shape != batch["content"].shape:
            raise ValueError("invalid BERT or source_index shapes")
        if torch.any(batch["content"] & ((source < 0) | (source >= bert.shape[-1]))):
            raise ValueError("content source_index outside BERT sequence")
        length = bert.shape[-1]
        audio = self._restore(batch["x"]["A"], batch["observed"]["A"], source, length)
        video = self._restore(batch["x"]["V"], batch["observed"]["V"], source, length)
        out = self.source(bert, (audio, None), (video, None))
        return {"raw_intensity": 3 * torch.tanh(out["M"].squeeze(-1) / 3),
                "aux_logits": self.classifier(out["Feature_f"])}


def build(config):
    model = config["model"]
    return TETFNSourceAdapter(**{key: model[key] for key in
                                 ("input_dims", "dst_width", "heads", "dropout",
                                  "post_fusion_dim", "pretrained_name") if key in model})
