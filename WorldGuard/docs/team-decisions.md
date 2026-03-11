# Team Decisions & Trade-offs (ADRs)

## ADR-001: ViT-S/16 as backbone (not ViT-B or ViT-L)

**Date**: 2026-03  
**Status**: Accepted

**Decision**: Use VideoMAE-pretrained ViT-S/16 as the video encoder.

**Reasoning**:
- ViT-S fits in 16GB VRAM at batch 16 with room to spare
- For anomaly *scoring* (not fine-grained classification), ViT-S representations are sufficient
- Faster iteration (8x faster per epoch than ViT-L) is more valuable at this stage
- Can always distill ViT-S → ViT-B if we need better AUROC numbers for publishing

**Trade-off**: Lower representation capacity may hurt on fine-grained anomaly types (e.g. distinguishing fighting from running).

---

## ADR-002: PyAV over OpenCV for video I/O

**Date**: 2026-03  
**Status**: Accepted

**Decision**: Use PyAV (Python bindings for FFmpeg) for all video reading.

**Reasoning**: OpenCV's VideoCapture has documented frame-seek errors on long recordings (>30 min). Surveillance footage is often hours long. PyAV seeks by keyframe correctly, giving reproducible clip extraction.

**Trade-off**: PyAV has a steeper API learning curve. Worth it for correctness.

---

## ADR-003: No HuggingFace Trainer / PyTorch Lightning

**Date**: 2026-03  
**Status**: Accepted

**Decision**: Raw PyTorch training loops only.

**Reasoning**: The EMA update must happen in a specific order relative to the optimizer step. Trainer abstractions obscure this and have caused subtle bugs in similar JEPA implementations. We want full visibility. Also, no Lightning dependency = fewer version conflicts on RTX A4000 CUDA setup.

---

## ADR-004: Per-camera thresholds (not global)

**Date**: 2026-03  
**Status**: Accepted

**Decision**: Every camera has its own anomaly threshold JSON in `configs/thresholds/`.

**Reasoning**: A parking lot camera and a server room camera have completely different "normal" score distributions. A global threshold would either flood the parking lot with false positives or miss real anomalies in the server room. Calibration is cheap (runs in minutes).

---

## Benchmark Results Log

| Date | Config | Dataset | Frame AUROC | Notes |
|---|---|---|---|---|
| — | — | — | — | Not yet run |
