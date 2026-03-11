# CLAUDE.md — data/

## What Lives Here
- `extract_clips.py` — Extracts 16-frame clips from raw MP4 using PyAV
- `dataset.py` — ClipDataset: loads .pt clips, splits into context/target, applies augmentations
- `augmentations.py` — Consistent spatio-temporal augmentations (same transform across all frames)

## Critical Rules

### Always Use PyAV for File I/O
`extract_clips.py` must use `av.open()` — NOT `cv2.VideoCapture`.
OpenCV has unreliable seek on long surveillance recordings (>30 min). PyAV is accurate.

### Augmentations Must Be Temporally Consistent
The same random crop and flip parameters must be applied to every frame in a clip.
Never sample new random params per frame — that destroys temporal structure.

```python
# CORRECT — sample once, apply to all T frames
i, j, h, w = RandomResizedCrop.get_params(clip[0], scale, ratio)
for t in range(T):
    frame = TF.crop(clip[t], i, j, h, w)

# WRONG — different crop per frame
for t in range(T):
    frame = random_crop(clip[t])  # breaks temporal coherence
```

### No Labels in this Directory
`dataset.py` returns `(context_frames, target_frames)` only.
No label loading, no anomaly flags. This is a self-supervised pipeline.

### Clip Tensor Format
All saved clips are `torch.Tensor` of shape `(clip_frames, 3, H, W)` in `float32 [0, 1]`.
Normalization (ImageNet mean/std) is applied in `dataset.py`, not in `extract_clips.py`.
