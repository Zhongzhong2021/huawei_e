import torch
from torch import nn


class SentimentModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.get("hidden", 64)
        self.encoder = config.get("encoder", "mean")
        self.fusion = config.get("fusion", "concat")
        self.embedding = nn.Embedding(30522, d, padding_idx=0)
        self.audio = nn.Linear(74, d)
        self.vision = nn.Linear(35, d)
        self.norm = nn.ModuleList([nn.LayerNorm(d) for _ in range(3)])
        if self.encoder == "gru":
            self.grus = nn.ModuleList([nn.GRU(d + 1, d // 2, batch_first=True, bidirectional=True) for _ in range(3)])
        if self.fusion == "gate":
            self.gate = nn.Sequential(nn.Linear(d + 1, 32), nn.Tanh(), nn.Linear(32, 1))
        self.head = nn.Sequential(nn.Linear(3 * d + 3, 128), nn.ReLU(),
                                  nn.Dropout(config.get("dropout", .2)), nn.Linear(128, 64), nn.ReLU())
        self.classifier = nn.Linear(64, 3)
        self.regressor = nn.Linear(64, 1)

    def forward(self, tokens, audio, vision, valid, observed, drop=None):
        obs = observed & valid.unsqueeze(-1)
        if drop is not None:
            obs = obs & ~drop
        # Critical: unseen text token IDs are replaced BEFORE embedding/GRU.
        tokens = tokens.masked_fill(~obs[:, :, 0], 0)
        audio = audio.masked_fill(~obs[:, :, 1, None], 0)
        vision = vision.masked_fill(~obs[:, :, 2, None], 0)
        features = [self.embedding(tokens), self.audio(audio), self.vision(vision)]
        pooled = []
        for m, x in enumerate(features):
            mask = obs[:, :, m, None]
            x = self.norm[m](x).masked_fill(~mask, 0)
            if self.encoder == "gru":
                x = torch.cat([x, mask.float()], dim=-1)
                # Only true positional extent enters bidirectional recurrence.
                lengths = (valid.long() * torch.arange(1, valid.shape[1] + 1, device=valid.device)).max(1).values.clamp_min(1)
                packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
                packed, _ = self.grus[m](packed)
                x, _ = nn.utils.rnn.pad_packed_sequence(packed, batch_first=True, total_length=valid.shape[1])
            h = (x * mask).sum(1) / mask.sum(1).clamp_min(1)
            pooled.append(h)
        h = torch.stack(pooled, 1)
        availability = obs.sum(1).float() / valid.sum(1, keepdim=True).clamp_min(1)
        if self.fusion == "gate":
            scores = self.gate(torch.cat([h, availability.unsqueeze(-1)], -1)).squeeze(-1)
            scores = scores.masked_fill(availability == 0, -1e4)
            weights = scores.softmax(1) * (availability > 0)
            weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-8)
            h = h * weights.unsqueeze(-1) * 3
        z = self.head(torch.cat([h.flatten(1), availability], 1))
        return self.classifier(z), 3 * torch.tanh(self.regressor(z).squeeze(-1))
