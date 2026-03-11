# WorldGuard — AI Briefing Doc

## What & Why
WorldGuard is a **JEPA-inspired video world model for unsupervised CCTV anomaly detection**.

The core thesis (from LeCun / AMI Labs): train a model to predict abstract latent representations
of normal scenes. Anomalies = high prediction error at inference. Zero labels required.

This is NOT a supervised classifier. We never train on anomaly examples. The model learns the
structure of normality and flags deviations.

## Stack
- Python 3.10+, PyTorch 2.x, CUDA 12.x
- Hardware: RTX A4000 (16GB VRAM), Ubuntu
- Key libs: timm, einops, opencv-python, wandb, av (PyAV for video I/O)
- No Hugging Face Trainer — raw PyTorch training loops only

## Where Things Live
```
WorldGuard/
├── configs/                    # YAML hyperparameter configs
│   ├── train_default.yaml      # Main training config
│   └── thresholds/             # Per-camera threshold JSONs (cam01.json, etc.)
├── data/                       # Dataset classes, clip extraction, augmentations
│   ├── extract_clips.py        # PyAV clip extractor (run first)
│   ├── dataset.py              # ClipDataset → (context_frames, target_frames)
│   └── augmentations.py        # ConsistentAugment, NormalizeVideo
├── models/                     # encoder.py, predictor.py, jepa_model.py
├── training/                   # train.py, calibrate.py, utils.py
├── inference/                  # score_video.py, heatmap.py, demo.py
├── eval/                       # ROC-AUC eval, t-SNE visualization
├── docs/                       # Architecture, runbooks, team decisions
├── checkpoints/                # Saved model weights (gitignored)
└── outputs/                    # Anomaly-flagged clips, heatmaps (gitignored)
```

## Core Rules
- **No pixel reconstruction** — all prediction happens in latent space only
- **No labels in training** — any code that loads anomaly labels during train/val is wrong
- **EMA encoder is always frozen** — never pass gradients through `target_encoder`
- **Batch size must fit 16GB** — default is 12-16 clips; add gradient checkpointing if OOM
- **Per-camera thresholds** — never use a global threshold; load from `configs/thresholds/{camera_id}.json`
- **Video I/O via PyAV** — not OpenCV VideoCapture (unreliable seek on long files)

## Anomaly Scoring — How It Works
1. Split clip: first 75% = context frames, last 25% = target frames
2. Context encoder → z_ctx embeddings
3. Predictor → z_pred (predicted future latent)
4. Target encoder (EMA, no grad) → z_tgt (actual future latent)
5. Anomaly score = mean L2(z_pred, z_tgt) across all patches
6. Compare against camera threshold → alert or pass

## Flow — How to Get Work Done
- **Adding a new model variant**: add to `models/`, register in `configs/train_default.yaml`
- **Running training**: `python training/train.py --config configs/train_default.yaml`
- **Calibrating thresholds**: `python training/calibrate.py --checkpoint <ckpt> --val-dir <dir>`
- **Scoring a video**: `python inference/score_video.py --video <path> --camera-id <id>`
- **Running eval**: `python eval/eval_roc.py --checkpoint <ckpt> --test-dir <dir>`

## State
- [x] Architecture design complete (see docs/architecture.md)
- [x] Training plan complete
- [x] Data pipeline — `extract_clips.py`, `dataset.py`, `augmentations.py`
- [ ] JEPA model implementation — `models/encoder.py`, `predictor.py`, `jepa_model.py`
- [ ] Training loop — `training/train.py`, `training/utils.py`
- [ ] Threshold calibration — `training/calibrate.py`
- [ ] Inference + heatmap — `inference/score_video.py`, `inference/heatmap.py`
- [ ] Evaluation on UCSD Ped2 — `eval/eval_roc.py`

## What Not To Do
- Do NOT use `torch.nn.DataParallel` — use single GPU only
- Do NOT use HuggingFace Trainer or Lightning — keep it raw PyTorch
- Do NOT hardcode paths — always use the config system
- Do NOT commit checkpoints or video files — they are gitignored
- Do NOT add dependencies without checking VRAM impact first

## Where to Look for More Context
- System design decisions → `docs/architecture.md`
- Operational procedures → `docs/runbooks.md`
- Key trade-offs we made → `docs/team-decisions.md`
- Code review checklist → `.claude/skills/code-review.md`
- Debugging flow → `.claude/skills/debugging-flow.md`
