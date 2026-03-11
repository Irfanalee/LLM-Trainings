# Architecture — WorldGuard

## System Overview

WorldGuard is a JEPA-style (Joint Embedding Predictive Architecture) world model applied to
unsupervised video anomaly detection. It is inspired by Meta's V-JEPA research and LeCun's
world model thesis (AMI Labs, 2026).

## Core Design Decisions

### Why JEPA over reconstruction-based methods?
Pixel reconstruction (MAE, generative models) is expensive and sensitive to irrelevant
variation (lighting flicker, leaves moving). JEPA predicts in abstract latent space — it
ignores unpredictable texture details and focuses on semantic, structural prediction.
Anomaly = semantic deviation, not pixel deviation. This is the right inductive bias for
surveillance.

### Why ViT-S/16 and not ViT-B or ViT-L?
ViT-S/16 (21M params) fits in 16GB VRAM at batch 16. ViT-B (~86M) is marginal. ViT-L (307M)
requires gradient checkpointing and batch 4, making training ~8x slower with minimal gain
for anomaly scoring (not fine-grained classification). Decision: ViT-S for fast iteration,
distill to ViT-B later if needed.

### Why EMA target encoder and not a separate network?
EMA prevents representation collapse without needing negative pairs (unlike contrastive methods).
The EMA copy slowly tracks the context encoder → stable training targets. This is the exact
mechanism used in V-JEPA, BYOL, and DINO.

### Why PyAV over OpenCV for video I/O?
OpenCV VideoCapture has unreliable seek behavior on long MP4 files (off-by-one frame errors
at high timestamps). PyAV seeks by keyframe accurately. Critical for reproducible clip extraction
from long surveillance recordings.

## Component Map

```
models/
  encoder.py       VideoMAE-pretrained ViT-S/16 + thin adapter layers for JEPA
  predictor.py     4-layer Transformer (D=384, heads=6) — learns world dynamics
  jepa_model.py    Combines encoder + predictor + EMA update logic

data/
  extract_clips.py  Extracts 16-frame clips from raw MP4 at given stride
  dataset.py        PyTorch Dataset — returns (context_frames, target_frames)
  augmentations.py  Consistent spatio-temporal augmentations across all frames in clip

training/
  train.py          Main loop — AdamW + cosine LR + EMA update + W&B logging
  calibrate.py      Runs val set → fits Gaussian → saves threshold JSON per camera

inference/
  score_video.py    Given video + camera_id → anomaly_score + is_anomaly + heatmap
  heatmap.py        Per-patch L2 errors → 14×14 grid → upsample to frame size
  demo.py           RTSP stream or webcam → live scoring overlay
```

## Data Flow

```
Raw CCTV MP4
    → extract_clips.py (stride=2, 16 frames, 224×224)
    → clips stored as .pt tensors or read on-the-fly
    → dataset.py: split clip → context (frames 0:12) + target (frames 12:16)
    → augmentations: RandomResizedCrop, HorizontalFlip (consistent across frames)
    → JEPAWorldModel.forward(context, target)
    → loss = L2(z_pred, z_tgt) in latent space
```

## VRAM Budget (RTX A4000, 16GB)

| Component | VRAM |
|---|---|
| Context encoder (ViT-S, with grads) | ~1.8 GB |
| Target encoder (EMA, no grads) | ~1.8 GB |
| Predictor transformer | ~0.4 GB |
| Activations (batch=16, 16 frames, 224×224) | ~8–10 GB |
| AdamW optimizer states | ~1.4 GB |
| **Total** | **~13–14 GB** |

## Known Limitations

- Per-camera thresholds must be recalibrated after major scene changes (camera angle, lighting)
- ViT-S/16 patch size = 16px — small fast-moving objects (<16px) may be missed
- Clip length = ~0.5s at 30fps — very slow anomalies (gradual gas leak) may be missed
