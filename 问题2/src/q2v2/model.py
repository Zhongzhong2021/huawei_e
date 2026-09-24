"""Small BERT initialized from local general pretraining; no transformers runtime.

Architecture follows BERT post-LayerNorm attention and GELU feed-forward blocks.
The checkpoint carries either the fitting-only compact vocabulary (unseen IDs
map to UNK) or the complete fixed pretrained vocabulary. Both load offline.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class BertBlock(nn.Module):
    def __init__(self, hidden=768, heads=12, intermediate=3072, dropout=.1):
        super().__init__()
        self.heads, self.head_size = heads, hidden // heads
        self.query, self.key, self.value = [nn.Linear(hidden, hidden) for _ in range(3)]
        self.attn_dense = nn.Linear(hidden, hidden)
        self.attn_norm = nn.LayerNorm(hidden, eps=1e-12)
        self.intermediate = nn.Linear(hidden, intermediate)
        self.out_dense = nn.Linear(intermediate, hidden)
        self.out_norm = nn.LayerNorm(hidden, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention):
        b, length, d = x.shape
        def split(layer):
            return layer(x).view(b, length, self.heads, self.head_size).transpose(1, 2)
        q, k, v = split(self.query), split(self.key), split(self.value)
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.head_size)
        scores = scores.masked_fill(~attention[:, None, None, :], -10000.)
        probabilities = self.dropout(scores.softmax(-1))
        context = (probabilities @ v).transpose(1, 2).contiguous().view(b, length, d)
        x = self.attn_norm(x + self.dropout(self.attn_dense(context)))
        return self.out_norm(x + self.dropout(self.out_dense(F.gelu(self.intermediate(x)))))


class CompactBert(nn.Module):
    def __init__(self, vocabulary, layers):
        super().__init__()
        vocabulary = torch.as_tensor(vocabulary, dtype=torch.long)
        self.register_buffer("vocabulary", vocabulary)
        lookup = torch.full((30522,), int((vocabulary == 100).nonzero()[0, 0]), dtype=torch.long)
        lookup[vocabulary] = torch.arange(len(vocabulary))
        self.register_buffer("lookup", lookup)
        self.word = nn.Embedding(len(vocabulary), 768, padding_idx=int((vocabulary == 0).nonzero()[0,0]))
        self.position = nn.Embedding(50, 768)
        self.segment = nn.Embedding(2, 768)
        self.emb_norm = nn.LayerNorm(768, eps=1e-12)
        self.dropout = nn.Dropout(.1)
        self.blocks = nn.ModuleList([BertBlock() for _ in layers])
        self.source_layers = list(layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0., std=.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
        if isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def initialize(self, source):
        mapped = {"word.weight": source["embeddings.word_embeddings.weight"][self.vocabulary],
                  "position.weight": source["embeddings.position_embeddings.weight"][:50],
                  "segment.weight": source["embeddings.token_type_embeddings.weight"],
                  "emb_norm.weight": source["embeddings.LayerNorm.weight"],
                  "emb_norm.bias": source["embeddings.LayerNorm.bias"]}
        mapping = {"query": "attention.self.query", "key": "attention.self.key", "value": "attention.self.value",
                   "attn_dense": "attention.output.dense", "attn_norm": "attention.output.LayerNorm",
                   "intermediate": "intermediate.dense", "out_dense": "output.dense", "out_norm": "output.LayerNorm"}
        for j, original in enumerate(self.source_layers):
            for dest, src in mapping.items():
                for part in ["weight", "bias"]:
                    mapped[f"blocks.{j}.{dest}.{part}"] = source[f"encoder.layer.{original}.{src}.{part}"]
        incompatible = self.load_state_dict(mapped, strict=False)
        assert set(incompatible.missing_keys) == {"lookup", "vocabulary"} and not incompatible.unexpected_keys

    def forward(self, tokens, attention):
        positions = torch.arange(tokens.shape[1], device=tokens.device)[None, :]
        x = self.word(self.lookup[tokens]) + self.position(positions) + self.segment(torch.zeros_like(tokens))
        x = self.dropout(self.emb_norm(x))
        for block in self.blocks:
            x = block(x, attention)
        return x


class PretrainedMultimodal(nn.Module):
    def __init__(self, config, vocabulary):
        super().__init__()
        self.config = dict(config)
        self.backbone = CompactBert(vocabulary, config["layers"])
        self.text_projection = nn.Sequential(nn.Linear(768, 128), nn.LayerNorm(128), nn.GELU())
        self.audio_projection = nn.Sequential(nn.Linear(74, 64), nn.LayerNorm(64), nn.GELU())
        self.vision_projection = nn.Sequential(nn.Linear(35, 64), nn.LayerNorm(64), nn.GELU())
        self.fusion = nn.Sequential(nn.Linear(259, 128), nn.GELU(), nn.Dropout(.2))
        self.classifier = nn.Linear(128, 3)
        self.regressor = nn.Linear(128, 1)

    def forward(self, tokens, audio, vision, valid, observed, drop=None, return_features=False):
        obs = observed & valid.unsqueeze(-1)
        if drop is not None:
            obs = obs & ~drop
        text_obs = obs[:, :, 0]
        special = (tokens == 101) | (tokens == 102)
        attention = text_obs | special
        attention = attention.clone()
        attention[:, 0] = True
        # Crucial: remove unavailable token IDs BEFORE contextual attention.
        safe_tokens = tokens.masked_fill(~attention, 0).clone()
        safe_tokens[:, 0] = 101
        contextual = self.backbone(safe_tokens, attention)
        pooled = (contextual * text_obs.unsqueeze(-1)).sum(1) / text_obs.sum(1, keepdim=True).clamp_min(1)
        text = self.text_projection(pooled) * text_obs.any(1, keepdim=True)
        av = []
        for index, raw, projector in [(1, audio, self.audio_projection), (2, vision, self.vision_projection)]:
            mask = obs[:, :, index, None]
            values = raw.masked_fill(~mask, 0)
            mean = values.sum(1) / mask.sum(1).clamp_min(1)
            h = projector(mean) * mask.any(1)
            av.append(h if self.config["multimodal"] else torch.zeros_like(h))
        availability = obs.sum(1).float() / valid.sum(1, keepdim=True).clamp_min(1)
        if not self.config["multimodal"]:
            availability = availability.clone()
            availability[:, 1:] = 0
        z = self.fusion(torch.cat([text, *av, availability], dim=1))
        logits, regression = self.classifier(z), 3 * torch.tanh(self.regressor(z).squeeze(-1))
        return (logits, regression, pooled) if return_features else (logits, regression)
