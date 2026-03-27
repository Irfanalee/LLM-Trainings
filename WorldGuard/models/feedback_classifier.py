"""Stage 2 feedback classifier for WorldGuard.

Takes z_pred embeddings from the JEPA model and outputs a probability
that a flagged clip is a true anomaly (vs. a false positive).

Trained on human-reviewed labels from training/review_anomalies.py.
The JEPA model is always frozen — this only trains the MLP head.
"""

import torch
import torch.nn as nn
from torch import Tensor


class FeedbackClassifier(nn.Module):
    """Lightweight MLP classifier on top of frozen JEPA z_pred embeddings.

    Args:
        embed_dim: Dimension of z_pred embeddings (default 384 for ViT-S).
        hidden_dim: Hidden layer size (default 128).
        dropout: Dropout probability (default 0.3).
    """

    def __init__(
        self,
        embed_dim: int = 384,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, z_pred: Tensor) -> Tensor:
        """Classify a batch of z_pred embeddings.

        Args:
            z_pred: (B, N, D) patch embeddings from JEPAWorldModel.

        Returns:
            (B,) probability of true anomaly in [0, 1].
        """
        x = z_pred.mean(dim=1)  # (B, N, D) → (B, D)
        return self.mlp(x).squeeze(-1)  # (B,)
