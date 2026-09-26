"""SELF-MM architecture and train-only pseudo-label adaptation for q3b.

Adapted from MMSA commit a94e65d, models/trains/multiTask/SELF_MM.py (MIT).
The q3b batch contract and extra competition classifier are local adaptations.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from .bert_init import load_bert


class AuViSubNet(nn.Module):
    """The one-layer LSTM, dropout, linear projection from MMSA SELF_MM."""

    def __init__(self, input_dim: int, hidden: int, output: int):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden, batch_first=True)
        self.dropout = nn.Dropout(0.0)
        self.linear_1 = nn.Linear(hidden, output)

    def forward(self, rows: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        result = rows.new_zeros((len(rows), self.linear_1.out_features))
        valid = lengths > 0
        if valid.any():
            packed = pack_padded_sequence(rows[valid], lengths[valid].cpu(),
                                          batch_first=True, enforce_sorted=False)
            _, (hidden, _) = self.rnn(packed)
            result[valid] = self.linear_1(self.dropout(hidden[-1]))
        return result


class SelfMMAdapter(nn.Module):
    local_deletion_is_exact = False
    native_explanation_scope = "no_native_attribution"

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        name = cfg.get("pretrained_name", "bert-base-uncased")
        self.text_model = load_bert(name)
        layers = self.text_model.encoder.layer
        self.trainable_layers = int(cfg.get("trainable_layers", 12))
        if not 0 <= self.trainable_layers <= len(layers):
            raise ValueError("invalid trainable_layers")
        for parameter in self.text_model.parameters():
            parameter.requires_grad_(False)
        for layer in list(layers)[-self.trainable_layers:] if self.trainable_layers else []:
            for parameter in layer.parameters():
                parameter.requires_grad_(True)
        text_dim = self.text_model.config.hidden_size
        audio_dim = int(cfg.get("audio_out", 16))
        video_dim = int(cfg.get("video_out", 32))
        fusion_dim = int(cfg.get("post_fusion_dim", 128))
        dims = {"T": int(cfg.get("post_text_dim", 64)), "A": int(cfg.get("post_audio_dim", 16)),
                "V": int(cfg.get("post_video_dim", 32))}
        self.audio_model = AuViSubNet(74, int(cfg.get("a_lstm_hidden_size", 16)), audio_dim)
        self.video_model = AuViSubNet(35, int(cfg.get("v_lstm_hidden_size", 32)), video_dim)
        self.post_fusion_dropout = nn.Dropout(float(cfg.get("post_fusion_dropout", .1)))
        self.post_fusion_layer_1 = nn.Linear(text_dim + audio_dim + video_dim, fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(fusion_dim, fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(fusion_dim, 1)
        self.post_dropout = nn.ModuleDict({m: nn.Dropout(float(cfg.get(f"post_{name}_dropout", 0)))
                                              for m, name in (("T", "text"), ("A", "audio"), ("V", "video"))})
        self.post_layer_1 = nn.ModuleDict({m: nn.Linear(in_dim, dims[m]) for m, in_dim in
                                           (("T", text_dim), ("A", audio_dim), ("V", video_dim))})
        self.post_layer_2 = nn.ModuleDict({m: nn.Linear(dims[m], dims[m]) for m in dims})
        self.post_layer_3 = nn.ModuleDict({m: nn.Linear(dims[m], 1) for m in dims})
        self.classification_weight = float(cfg.get("classification_weight", 0))
        self.aux_head = nn.Linear(fusion_dim, 3) if self.classification_weight else None
        self.feature_dims = {"M": fusion_dim, **dims}
        self.H = float(cfg.get("H", 3.0))
        self.label_map = None
        self.feature_map = None
        self.center_map = None
        self.seen = None
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, "text_model"):
            self.text_model.embeddings.eval()
            for layer in list(self.text_model.encoder.layer)[:-self.trainable_layers or None]:
                layer.eval()
            if getattr(self.text_model, "pooler", None) is not None:
                self.text_model.pooler.eval()
        return self

    @staticmethod
    def _ordered_rows(batch: dict, modality: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress observed rows in original source order; padding stays after them."""
        x = batch["x"][modality].flatten(1, 2)
        observed = batch["observed"][modality].flatten(1, 2)
        source = batch["source_index"].flatten(1, 2)
        rows = torch.zeros_like(x)
        lengths = observed.sum(1)
        for i in range(len(x)):
            positions = torch.where(observed[i])[0]
            order = source[i, positions].argsort(stable=True)
            rows[i, :len(positions)] = x[i, positions[order]]
        return rows, lengths

    def forward(self, batch: dict, interaction_gain: float = 1.0) -> dict:
        bert = batch["bert_inputs"]
        text = self.text_model(input_ids=bert[:, 0].long(), attention_mask=bert[:, 1].long(),
                               token_type_ids=bert[:, 2].long()).last_hidden_state[:, 0]
        audio, a_len = self._ordered_rows(batch, "A")
        vision, v_len = self._ordered_rows(batch, "V")
        reps = {"T": text, "A": self.audio_model(audio, a_len),
                "V": self.video_model(vision, v_len)}
        fused = torch.cat((reps["T"], reps["A"], reps["V"]), dim=-1)
        feature_m = F.relu(self.post_fusion_layer_1(self.post_fusion_dropout(fused)))
        features = {"M": feature_m}
        outputs = {"M": self.post_fusion_layer_3(F.relu(self.post_fusion_layer_2(feature_m))).squeeze(-1)}
        for modality in ("T", "A", "V"):
            feature = F.relu(self.post_layer_1[modality](self.post_dropout[modality](reps[modality])))
            features[modality] = feature
            outputs[modality] = self.post_layer_3[modality](F.relu(self.post_layer_2[modality](feature))).squeeze(-1)
        raw = outputs["M"]
        # The source regression-only control has no trained classifier.
        logits = self.aux_head(feature_m) if self.aux_head is not None else torch.stack(
            (-raw, -raw.abs(), raw), dim=1)
        return {"raw_intensity": raw, "aux_logits": logits,
                "selfmm_outputs": outputs, "selfmm_features": features}

    def optimizer_groups(self, cfg: dict) -> list[dict]:
        groups = {}
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("text_model."):
                lr = cfg.get("encoder_learning_rate", cfg["learning_rate"])
                decay = cfg.get("encoder_weight_decay", cfg.get("weight_decay", .01))
                if parameter.ndim < 2:
                    decay = 0.0
            elif name.startswith("audio_model."):
                lr = cfg.get("audio_learning_rate", cfg["learning_rate"])
                decay = cfg.get("audio_weight_decay", cfg.get("weight_decay", .01))
            elif name.startswith("video_model."):
                lr = cfg.get("video_learning_rate", cfg["learning_rate"])
                decay = cfg.get("video_weight_decay", cfg.get("weight_decay", .01))
            else:
                lr = cfg.get("other_learning_rate", cfg["learning_rate"])
                decay = cfg.get("weight_decay", .01)
            groups.setdefault((lr, decay), []).append(parameter)
        return [dict(params=parameters, lr=lr, weight_decay=decay)
                for (lr, decay), parameters in groups.items()]

    def prepare_training(self, train_batch: dict) -> None:
        index = train_batch["sample_index"].long()
        if index.ndim != 1 or len(index) != len(train_batch["y"]) or index.min() < 0:
            raise ValueError("training sample_index must be nonnegative and match labels")
        if index.unique().numel() != len(index):
            raise ValueError("training sample_index must be unique")
        size = int(index.max()) + 1
        labels = train_batch["y"].detach()
        self.label_map = {m: torch.zeros(size, device=labels.device) for m in self.feature_dims}
        self.feature_map = {m: torch.zeros(size, dim, device=labels.device) for m, dim in self.feature_dims.items()}
        self.center_map = {m: {"pos": torch.zeros(dim, device=labels.device),
                               "neg": torch.zeros(dim, device=labels.device)}
                           for m, dim in self.feature_dims.items()}
        self.seen = torch.zeros(size, dtype=torch.bool, device=labels.device)
        self.train_index = torch.zeros(size, dtype=torch.bool, device=labels.device)
        self.train_index[index] = True
        for values in self.label_map.values():
            values[index] = labels

    def _indices(self, batch: dict) -> torch.Tensor:
        if self.label_map is None:
            raise RuntimeError("prepare_training must run before training")
        index = batch["sample_index"].long()
        if index.ndim != 1 or index.min() < 0 or index.max() >= len(self.train_index) or not self.train_index[index].all():
            raise ValueError("pseudo-label updates require training sample_index only")
        return index

    def training_loss(self, out: dict, batch: dict, epoch: int) -> torch.Tensor:
        index = self._indices(batch)
        loss = F.l1_loss(out["selfmm_outputs"]["M"], batch["y"])
        for modality in ("T", "A", "V"):
            weight = torch.tanh((self.label_map[modality][index] - self.label_map["M"][index]).abs())
            loss = loss + (weight * (out["selfmm_outputs"][modality] - self.label_map[modality][index]).abs()).mean()
        if self.classification_weight:
            loss = loss + self.classification_weight * F.cross_entropy(out["aux_logits"], batch["class_label"])
        return loss

    @torch.no_grad()
    def after_train_batch(self, out: dict, batch: dict, epoch: int) -> None:
        index = self._indices(batch)
        features = {m: x.detach() for m, x in out["selfmm_features"].items()}
        if epoch > 1 and self.seen.any():
            center = self.center_map
            df = ((features["M"] - center["M"]["neg"]).norm(dim=1)
                  - (features["M"] - center["M"]["pos"]).norm(dim=1))
            df = df / (features["M"] - center["M"]["pos"]).norm(dim=1).clamp_min(1e-8)
            safe_df = torch.where(df.abs() < 1e-5, torch.where(df < 0, -1e-5, 1e-5), df)
            for modality in ("T", "A", "V"):
                ds = ((features[modality] - center[modality]["neg"]).norm(dim=1)
                      - (features[modality] - center[modality]["pos"]).norm(dim=1))
                ds = ds / (features[modality] - center[modality]["pos"]).norm(dim=1).clamp_min(1e-8)
                alpha = (ds / safe_df).clamp(-10, 10)
                target = (.5 * alpha * self.label_map["M"][index]
                          + .5 * (self.label_map["M"][index] + ds - df)).clamp(-self.H, self.H)
                self.label_map[modality][index] = ((epoch - 1) / (epoch + 1) * self.label_map[modality][index]
                                                    + 2 / (epoch + 1) * target)
        for modality in self.feature_dims:
            self.feature_map[modality][index] = features[modality]
        self.seen[index] = True
        for modality in self.feature_dims:
            labels = self.label_map[modality]
            for sign, mask in (("pos", labels > 0), ("neg", labels < 0)):
                selected = self.seen & self.train_index & mask
                if selected.any():
                    self.center_map[modality][sign] = self.feature_map[modality][selected].mean(0)


def build(config: dict) -> SelfMMAdapter:
    return SelfMMAdapter(config["model"])
