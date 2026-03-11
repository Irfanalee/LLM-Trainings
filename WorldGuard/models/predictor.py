"""JEPAPredictor — 4-layer Transformer that predicts target latents from context latents.

Architecture:
    1. Temporal pool: mean-aggregate T_ctx frame embeddings → (B, N, D)
       (mean pooling is appropriate for mostly-static surveillance scenes)
    2. Add learnable positional embeddings (one per patch position)
    3. 4-layer pre-norm Transformer: spatial reasoning over the 196 patch tokens
    4. Output: (B, N, D) predicted embeddings to compare against target encoder

Sequence length is N=196 (not T*N), keeping attention cost at O(196²) per layer
instead of O(2352²) — critical for staying within the 16GB VRAM budget.
"""

import torch
import torch.nn as nn


class JEPAPredictor(nn.Module):
    """Predicts future scene latents from context frame embeddings.

    Args:
        config: Loaded YAML config dict.
    """

    NUM_PATCHES = 196  # fixed by ViT-S/16 on 224×224 input: (224/16)²

    def __init__(self, config: dict) -> None:
        super().__init__()

        cfg = config["model"]
        dim: int = cfg["encoder_dim"]        # 384
        depth: int = cfg["predictor_layers"] # 4
        heads: int = cfg["predictor_heads"]  # 6
        mlp_dim: int = dim * 4               # 1536 — standard Transformer ratio

        # Learnable positional embedding (per patch position, shared across time)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.NUM_PATCHES, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Pre-norm Transformer (more training-stable than post-norm)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=0.0,       # no dropout — model relies on EMA for regularization
            activation="gelu",
            batch_first=True,
            norm_first=True,   # pre-norm
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)

    def forward(self, z_ctx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_ctx: Context encoder output (B, T_ctx, N, D).

        Returns:
            Predicted target patch embeddings (B, N, D).
        """
        # Aggregate context frames — mean over temporal dim
        x = z_ctx.mean(dim=1)    # (B, N, D)

        # Inject spatial position information
        x = x + self.pos_embed   # (B, 196, D)

        # Transformer: spatial reasoning over patch tokens
        x = self.transformer(x)  # (B, 196, D)
        x = self.norm(x)

        return x  # (B, N, D)
