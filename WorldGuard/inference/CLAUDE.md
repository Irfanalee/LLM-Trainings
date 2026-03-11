# CLAUDE.md — inference/

## Purpose
This directory runs the trained model on new footage to produce anomaly scores and heatmaps.

## Key Contracts

### score_video.py — Always Return Both Score and Heatmap
The function signature must always be:
```python
def score_clip(model, frames, threshold) -> (float, bool, np.ndarray):
    # Returns: (anomaly_score, is_anomaly, heatmap_224x224)
```
Never return just `is_anomaly`. The raw score and spatial heatmap are required for debugging
and for the LinkedIn demo visualization.

### Thresholds Are Always Camera-Specific
```python
# CORRECT
threshold = load_threshold("configs/thresholds/cam01.json")

# WRONG — never hardcode or use a global default
threshold = 0.05
```

### heatmap.py — Normalization
Heatmaps must be normalized to [0, 1] before returning. Raw L2 values have no consistent scale
across cameras and checkpoints.

```python
heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
```

### demo.py — RTSP Support
`demo.py` should support both:
- `--source webcam` (device 0)
- `--source rtsp://...` (IP camera stream via PyAV)

OpenCV VideoCapture is acceptable here (it's a live stream, not a long file, so seek issues don't apply).
