"""SELF-MM temporal/center transfer with source-aligned additive readout.

Not a source reproduction: per-row LSTM outputs replace final-state pooling;
train-only masked centers supervise native unimodal scores. Readouts remain
contextual, not independent causal source effects.
"""
from __future__ import annotations

import torch
from torch import nn
from .model import MODALITIES, PAIRS, _additive_output
from .native_next import NativeNextModel
from .selfmm_adapter import SelfMMAdapter


class NativeSelfMM(NativeNextModel):
    def __init__(self, *, temporal_dim=32, pseudo_weight=0., **kwargs):
        dims = kwargs['input_dims']
        super().__init__(**{**kwargs, 'input_dims': {**dims, 'A': temporal_dim, 'V': temporal_dim}})
        self.temporal = nn.ModuleDict({m: nn.LSTM(dims[m], temporal_dim, batch_first=True)
                                      for m in ('A', 'V')})
        self.pseudo_weight = float(pseudo_weight)
        if self.pseudo_weight < 0:
            raise ValueError('pseudo_weight must be nonnegative')
        self.feature_dims = {m: self.hidden_dim for m in MODALITIES}
        self.feature_dims['M'] = 3 * self.hidden_dim
        self.H = 3.
        self.label_map = self.feature_map = self.center_map = self.seen = None
        self.pseudo_audit = []

    def temporal_rows(self, batch, m):
        x = batch['x'][m].flatten(1, 2)
        observed = batch['observed'][m].flatten(1, 2)
        source = batch['source_index'].flatten(1, 2)
        content = batch['content'].flatten(1, 2)
        # Sort all content rows, including internally missing observations.
        order = source.masked_fill(~content, torch.iinfo(source.dtype).max).argsort(dim=1, stable=True)
        ordered = (x * observed.unsqueeze(-1)).gather(1, order.unsqueeze(-1).expand_as(x))
        hidden, _ = self.temporal[m](ordered)
        restored = torch.zeros_like(hidden).scatter(1, order.unsqueeze(-1).expand_as(hidden), hidden)
        return (restored * observed.unsqueeze(-1)).reshape(*batch['observed'][m].shape, -1)

    def forward(self, batch, interaction_gain=1.):
        text = self._text_rows(batch)
        d, weights, valid, features = {}, {}, {}, {}
        for m in MODALITIES:
            x = text if m == 'T' else self.temporal_rows(batch, m)
            observed = batch['observed'][m]
            if m == 'T' and 'bert_readout_mask' in batch:
                x = x * batch['bert_readout_mask'].unsqueeze(-1)
            encoder = self.encoders[m]
            delta = encoder(x, observed) - encoder(torch.zeros_like(x), observed)
            d[m] = self.dropout(delta) * observed.any(-1, keepdim=True)
            weights[m], valid[m] = self._weights(d[m], observed, m)
            features[m] = (d[m] * weights[m].unsqueeze(-1)).sum(1)
        single = torch.stack([self.single_heads[m](d[m]) * weights[m].unsqueeze(-1)
                              for m in MODALITIES], 2)
        pairs = []
        for a, b in PAIRS:
            joint = (weights[a] * weights[b]).clamp_min(1e-20).sqrt() * (valid[a] & valid[b])
            joint = joint / joint.sum(1, keepdim=True).clamp_min(1e-12)
            pairs.append(self.pair_modules[a+b](d[a], d[b]) * joint.unsqueeze(-1) * interaction_gain)
        pair = torch.stack(pairs, 2)
        available = torch.stack([valid[m].any(1) for m in MODALITIES], 1)
        unimodal = single.sum(1) + self.bias
        keep = self._modality_keep(available)
        pair_keep = torch.stack([keep[:, MODALITIES.index(a)] & keep[:, MODALITIES.index(b)]
                                 for a, b in PAIRS], 1)
        out = _additive_output(single * keep[:, None, :, None], pair * pair_keep[:, None, :, None], self.bias)
        features['M'] = torch.cat([features[m] for m in MODALITIES], -1)
        out.update(unimodal_scores=unimodal, modality_available=available, modality_keep=keep,
                   selfmm_features=features)
        return out

    def prepare_training(self, train_batch):
        if not self.training:
            raise RuntimeError('training mode required for pseudo-label initialization')
        SelfMMAdapter.prepare_training(self, train_batch)
        self.observed_map = {m: torch.zeros_like(self.seen) for m in self.feature_dims}

    def _indices(self, batch):
        if not self.training:
            raise RuntimeError('pseudo-label state can only be used in training mode')
        index = SelfMMAdapter._indices(self, batch)
        if len(index) != len(batch['y']) or len(index.unique()) != len(index):
            raise ValueError('training sample_index must match batch and be unique')
        if not torch.equal(batch['y'], self.label_map['M'][index]):
            raise ValueError('sample_index labels differ from initialized training labels')
        return index

    def training_loss(self, out, batch, epoch):
        loss = super().training_loss(out, batch, epoch)
        if self.pseudo_weight:
            index = self._indices(batch)
            terms = []
            for j, m in enumerate(MODALITIES):
                target = self.label_map[m][index]
                weight = torch.tanh((target - self.label_map['M'][index]).abs())
                mask = out['modality_available'][:, j]
                error = (3 * torch.tanh(out['unimodal_scores'][:, j, 0]) - target).abs()
                terms.append((weight * error * mask).sum() / mask.sum().clamp_min(1))
            loss = loss + self.pseudo_weight * sum(terms)
        return loss

    @torch.no_grad()
    def after_train_batch(self, out, batch, epoch):
        index = self._indices(batch)
        if not self.pseudo_weight:
            return
        features = {m: f.detach() for m, f in out['selfmm_features'].items()}
        masks = {m: out['modality_available'][:, j] for j, m in enumerate(MODALITIES)}
        masks['M'] = out['modality_available'].any(1)
        def distance(m):
            pos = (features[m] - self.center_map[m]['pos']).norm(dim=1)
            neg = (features[m] - self.center_map[m]['neg']).norm(dim=1)
            return (neg-pos) / pos.clamp_min(1e-8)
        if epoch > 1 and self.seen.any():
            df = distance('M')
            safe = torch.where(df.abs() < 1e-5, torch.where(df < 0, -1e-5, 1e-5), df)
            for m in MODALITIES:
                ds = distance(m)
                target = (.5 * (ds/safe).clamp(-10, 10) * self.label_map['M'][index]
                          + .5 * (self.label_map['M'][index] + ds-df)).clamp(-self.H, self.H)
                ids = index[masks[m]]
                self.label_map[m][ids] = ((epoch-1)/(epoch+1) * self.label_map[m][ids]
                                          + 2/(epoch+1) * target[masks[m]])
        self.seen[index] = True
        for m in self.feature_dims:
            self.feature_map[m][index] = features[m]
            self.observed_map[m][index] = masks[m]
            for sign, sign_mask in (('pos', self.label_map[m] > 0), ('neg', self.label_map[m] < 0)):
                selected = self.seen & self.train_index & self.observed_map[m] & sign_mask
                if selected.any():
                    self.center_map[m][sign] = self.feature_map[m][selected].mean(0)

    def after_train_epoch(self, epoch):
        if self.pseudo_weight:
            audit = {'epoch': epoch, 'pseudo_mean_delta': {
                m: float((self.label_map[m][self.observed_map[m]] - self.label_map['M'][self.observed_map[m]]).abs().mean())
                for m in MODALITIES}}
            self.pseudo_audit.append(audit)
            print(audit, flush=True)


def build(config):
    keys = ('input_dims', 'row_dim', 'hidden_dim', 'rank', 'window_size', 'dropout',
            'trainable_layers', 'pooling', 'pretrained_name', 'modality_dropout', 'temporal_dim', 'pseudo_weight')
    return NativeSelfMM(**{k: config['model'][k] for k in keys if k in config['model']},
                        train_config=config.get('train', {}))
