"""JEPAWorldModel — core model: context encoder + predictor + EMA target encoder.

Training flow (order is non-negotiable — see models/CLAUDE.md):
    1. z_ctx  = context_encoder(context)         # grads flow
    2. z_pred = predictor(z_ctx)                 # grads flow
    3. z_tgt  = target_encoder(target)           # NO grads — torch.no_grad()
    4. loss   = MSE(z_pred, z_tgt)
    5. loss.backward(); optimizer.step()
    6. model.update_target_encoder()             # EMA — ALWAYS after optimizer.step()

CRITICAL: target_encoder must NEVER receive gradients.
Breaking this causes silent collapse — loss decreases but scores are garbage.
"""

import copy
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import VideoEncoder
from models.predictor import JEPAPredictor


class JEPAOutput(NamedTuple):
    loss: torch.Tensor    # scalar MSE loss
    z_pred: torch.Tensor  # (B, N, D) — predicted patch embeddings
    z_tgt: torch.Tensor   # (B, N, D) — target patch embeddings (for heatmap)


class JEPAWorldModel(nn.Module):
    """Joint Embedding Predictive Architecture for video anomaly detection.

    The model learns to predict the latent representation of future frames given
    context frames. At inference, high prediction error = anomaly.

    Args:
        config: Loaded YAML config dict.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()

        self.ema_decay: float = config["model"]["ema_decay"]  # 0.996

        # Context encoder — trained by optimizer
        self.context_encoder = VideoEncoder(config, pretrained=True)

        # Target encoder — EMA copy, never trained directly
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad_(False)

        self.predictor = JEPAPredictor(config)

    def forward(self, context: torch.Tensor, target: torch.Tensor) -> JEPAOutput:
        """Compute JEPA loss for a batch of clips.

        Args:
            context: (B, T_ctx, C, H, W) — context frames (first 75% of clip).
            target:  (B, T_tgt, C, H, W) — target frames (last 25% of clip).

        Returns:
            JEPAOutput(loss, z_pred, z_tgt).
        """
        # Context path — gradients flow through here
        z_ctx = self.context_encoder(context)  # (B, T_ctx, N, D)
        z_pred = self.predictor(z_ctx)          # (B, N, D)

        # Target path — no gradients, ever
        with torch.no_grad():
            z_tgt_frames = self.target_encoder(target)  # (B, T_tgt, N, D)
            z_tgt = z_tgt_frames.mean(dim=1)            # (B, N, D) temporal pool

        loss = F.mse_loss(z_pred, z_tgt)

        return JEPAOutput(loss=loss, z_pred=z_pred, z_tgt=z_tgt)

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        """EMA update: target = ema_decay * target + (1 - ema_decay) * context.

        Call this AFTER optimizer.step(), never before. See models/CLAUDE.md.
        """
        for ctx_p, tgt_p in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            tgt_p.data.mul_(self.ema_decay).add_(ctx_p.data, alpha=1.0 - self.ema_decay)

    @torch.no_grad()
    def patch_errors(self, context: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-patch L2 prediction error for heatmap generation.

        Used by inference/heatmap.py. Returns unnormalized errors;
        normalization to [0, 1] happens in heatmap.py.

        Args:
            context: (B, T_ctx, C, H, W)
            target:  (B, T_tgt, C, H, W)

        Returns:
            patch_errors: (B, N) — per-patch MSE across embedding dim D.
        """
        z_ctx = self.context_encoder(context)
        z_pred = self.predictor(z_ctx)
        z_tgt_frames = self.target_encoder(target)
        z_tgt = z_tgt_frames.mean(dim=1)

        return (z_pred - z_tgt).pow(2).mean(dim=-1)  # (B, N)
