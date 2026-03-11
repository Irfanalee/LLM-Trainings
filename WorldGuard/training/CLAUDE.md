# CLAUDE.md — training/

## What Lives Here
- `train.py` — Main training loop
- `calibrate.py` — Threshold calibration on validation set
- `utils.py` — EMA update helper, W&B logging, checkpoint save/load

## Critical Constraints

### Labels Are Forbidden in train.py
The training loop must NEVER load or use anomaly labels. If you see label loading here, it's a bug.
Training is 100% self-supervised. Labels only appear in `eval/`.

### W&B Logging — Required Metrics
Every training run MUST log these to W&B:
- `train/loss` — per step
- `train/lr` — per step
- `gpu/memory_mb` — per step (use `torch.cuda.memory_allocated() / 1e6`)
- `val/mean_score` — per epoch (average prediction error on val set)
- `val/std_score` — per epoch (used for threshold calibration)

Missing metrics make experiment comparison impossible. Don't skip them.

### Checkpoint Naming Convention
```
checkpoints/
  {config_name}_epoch{N}_auroc{score}.pt
  e.g.: vits16_bs16_lr1e4_epoch50_auroc0.83.pt
```

### calibrate.py Must Not Modify the Checkpoint
Calibration is read-only. It writes ONLY to `configs/thresholds/{camera_id}.json`.
It does NOT fine-tune the model.
