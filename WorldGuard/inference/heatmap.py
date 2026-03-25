"""Heatmap generation for WorldGuard anomaly visualization.

Converts per-patch L2 prediction errors from the JEPA model into a
spatial heatmap overlaid on the video frame.

Pipeline:
    patch_errors (B, 196) → reshape (B, 14, 14) → normalize [0,1]
    → upsample (B, 224, 224) → overlay on frame (optional)
"""

import numpy as np
import torch
import torch.nn.functional as F

# ViT-S/16 on 224×224 produces a 14×14 patch grid
PATCH_GRID = 14
NUM_PATCHES = PATCH_GRID * PATCH_GRID  # 196


def generate_heatmap(
    model,
    context: torch.Tensor,
    target: torch.Tensor,
    frame_size: int = 224,
) -> np.ndarray:
    """Generate a spatial anomaly heatmap for one clip.

    Args:
        model: JEPAWorldModel in eval mode.
        context: (T_ctx, C, H, W) or (1, T_ctx, C, H, W) — context frames.
        target:  (T_tgt, C, H, W) or (1, T_tgt, C, H, W) — target frames.
        frame_size: Output heatmap spatial resolution (default 224).

    Returns:
        heatmap: (frame_size, frame_size) float32 in [0, 1].
                 Higher values = higher anomaly likelihood.
    """
    # Ensure batch dimension
    if context.dim() == 4:
        context = context.unsqueeze(0)
    if target.dim() == 4:
        target = target.unsqueeze(0)

    with torch.no_grad():
        errors = model.patch_errors(context, target)  # (1, 196)

    # Reshape to spatial grid
    grid = errors.view(1, 1, PATCH_GRID, PATCH_GRID)  # (1, 1, 14, 14)

    # Upsample to frame_size
    grid_up = F.interpolate(
        grid, size=(frame_size, frame_size), mode="bilinear", align_corners=False
    )  # (1, 1, H, W)

    heatmap = grid_up.squeeze().cpu().float().numpy()  # (H, W)

    # Normalize to [0, 1]
    lo, hi = heatmap.min(), heatmap.max()
    if hi > lo:
        heatmap = (heatmap - lo) / (hi - lo)
    else:
        heatmap = np.zeros_like(heatmap)

    return heatmap


def overlay_heatmap(
    frame: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay a heatmap on a video frame using a red colormap.

    Args:
        frame:   (H, W, 3) uint8 RGB frame.
        heatmap: (H, W) float32 in [0, 1].
        alpha:   Blend weight for the heatmap overlay (0 = frame only, 1 = heatmap only).

    Returns:
        blended: (H, W, 3) uint8 RGB image.
    """
    # Red colormap: map heatmap intensity to red channel
    heat_rgb = np.zeros((*heatmap.shape, 3), dtype=np.float32)
    heat_rgb[..., 0] = heatmap        # red channel
    heat_rgb[..., 1] = 0.0            # no green
    heat_rgb[..., 2] = 0.0            # no blue

    frame_f = frame.astype(np.float32) / 255.0
    blended = (1.0 - alpha) * frame_f + alpha * heat_rgb
    blended = np.clip(blended * 255.0, 0, 255).astype(np.uint8)
    return blended
