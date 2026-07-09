"""ExoplanetTransformer model architecture.

Extracted from transformer.ipynb so it can be imported by the simulation-based
pretraining notebooks (pretrain_sim.ipynb, pretrain_stage2.ipynb,
finetune_adversarial.ipynb, eval_sim_pretrained.ipynb).

Architecture:
    input_proj: Linear(21 -> 48)
    encoder:    1-layer TransformerEncoder (4 heads, d_model=48, ff=96, dropout=0.3)
    pool:       AttentionPool (learned attention over observations)
    classifier: Linear(48 -> 16) -> ReLU -> Dropout -> Linear(16 -> 1)

Input:  (batch, seq_len, 21) — 5 raw features + 16 sinusoidal timestamp dims
Output: (batch,) — raw classification logits (use BCEWithLogitsLoss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPool(nn.Module):
    """Learn which observations matter most for the classification.

    Defined before ExoplanetTransformer so it can be referenced as a
    module-level class by all importing notebooks.
    """
    def __init__(self, d_model):
        super().__init__()
        self.attention = nn.Linear(d_model, 1)

    def forward(self, x, mask):
        scores = self.attention(x).squeeze(-1)  # (B, seq_len)
        scores = scores.masked_fill(~mask.bool(), float('-inf'))
        weights = F.softmax(scores, dim=1)  # (B, seq_len)
        pooled = (x * weights.unsqueeze(-1)).sum(dim=1)  # (B, d_model)
        return pooled


class ExoplanetTransformer(nn.Module):
    def __init__(self, feat_dim=21, d_model=48, nhead=4, num_layers=1,
                 dim_feedforward=96, dropout=0.3):
        super().__init__()

        self.input_proj = nn.Linear(feat_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pool = AttentionPool(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, x, mask):
        x = self.input_proj(x)
        src_key_padding_mask = ~mask.bool()
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.pool(x, mask)
        out = self.classifier(x).squeeze(-1)
        return out
