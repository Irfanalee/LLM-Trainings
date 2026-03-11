"""VideoEncoder — ViT-S/16 backbone with thin adapter for JEPA.

This class is instantiated twice inside JEPAWorldModel:
  - context_encoder: receives gradients, updated by optimizer
  - target_encoder:  EMA copy, param.requires_grad=False, updated by EMA only

Processing: each frame is passed through ViT independently.
  (B, T, C, H, W) → flatten to (B*T, C, H, W) → ViT → reshape to (B, T, N, D)

Patch count N = (224 / 16)² = 196, embedding dim D = 384 (ViT-S).
"""

import torch
import torch.nn as nn
import timm


class VideoEncoder(nn.Module):
    """Encodes video frames into per-patch latent embeddings.

    Args:
        config: Loaded YAML config dict.
        pretrained: Load ImageNet pretrained ViT weights. Set True for the
                    context encoder init; the target encoder is deep-copied
                    from the context encoder in jepa_model.py.
    """

    def __init__(self, config: dict, pretrained: bool = True) -> None:
        super().__init__()

        embed_dim: int = config["model"]["encoder_dim"]  # 384

        # ViT-S/16 — global_pool='' returns all tokens (CLS + patches), no head
        self.vit = timm.create_model(
            "vit_small_patch16_224",
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )

        # Derived from the model so we don't hardcode 196 anywhere
        self.num_patches: int = self.vit.patch_embed.num_patches  # 196

        # Optional gradient checkpointing (controlled by config to save VRAM)
        if config["training"].get("grad_checkpoint", False):
            self.vit.set_grad_checkpointing(enable=True)

        # Thin adapter: LayerNorm + linear projection
        # Kept small — a heavier adapter would blow the VRAM budget
        self.adapter = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim, bias=False),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: (B, T, C, H, W) — batch of video clips.

        Returns:
            Patch embeddings of shape (B, T, N, D) where N=196, D=384.
        """
        B, T, C, H, W = frames.shape

        # Process all frames in one pass through ViT
        x = frames.view(B * T, C, H, W)

        # forward_features returns (B*T, N+1, D): index 0 = CLS, 1: = patches
        features = self.vit.forward_features(x)  # (B*T, 197, 384)
        patch_tokens = features[:, 1:, :]         # (B*T, 196, 384) — drop CLS

        patch_tokens = self.adapter(patch_tokens)  # (B*T, 196, 384)

        D = patch_tokens.shape[-1]
        return patch_tokens.view(B, T, self.num_patches, D)  # (B, T, 196, 384)
