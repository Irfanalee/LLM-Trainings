# Skill: Experiment & Release Procedure

Follow this whenever you're running a new training experiment or releasing a model checkpoint.

## Before Starting a Training Run

- [ ] Config file is committed to git with a meaningful name (e.g. `vits16_bs16_lr1e4_ep100.yaml`)
- [ ] W&B run name matches config filename
- [ ] `nvidia-smi` shows expected free VRAM (should be > 14GB before run)
- [ ] Val set is confirmed separate from train set (different recording dates/sessions)
- [ ] Previous checkpoint path is set in config if continuing training

## During Training — What to Monitor (W&B)

| Metric | Expected behaviour | Action if wrong |
|---|---|---|
| `train/loss` | Decreasing, no NaN | Stop, debug (see debugging-flow.md) |
| `train/lr` | Warmup then cosine decay | Check scheduler config |
| `gpu/memory_used` | Stable, < 15.5GB | Reduce batch size if growing |
| `model/ema_decay` | Fixed at 0.996 | Check EMA update is called |
| `val/mean_score` | Stable after epoch 10 | If rising: possible data issue |

## After Training — Checkpoint Validation

```bash
# 1. Run calibration on validation set
python training/calibrate.py \
  --checkpoint checkpoints/best.pt \
  --val-dir data/val_normal/ \
  --camera-id cam01 \
  --output configs/thresholds/cam01.json

# 2. Quick sanity check on known normal clip (score should be LOW)
python inference/score_video.py \
  --video data/test_normal/sample.mp4 \
  --camera-id cam01

# 3. Quick sanity check on known anomaly clip (score should be HIGH)
python inference/score_video.py \
  --video data/test_anomaly/fire_sample.mp4 \
  --camera-id cam01

# 4. Run full eval (only if sanity checks pass)
python eval/eval_roc.py \
  --checkpoint checkpoints/best.pt \
  --test-dir data/ucsd_ped2/ \
  --output outputs/eval_results.json
```

## Publishing a Checkpoint to GitHub

- [ ] ROC-AUC is documented in `docs/team-decisions.md` with config used
- [ ] Checkpoint is NOT committed (gitignored) — link to external storage in README
- [ ] README updated with new benchmark numbers
- [ ] W&B run link added to README
- [ ] Any architecture changes are documented in `docs/architecture.md`

## LinkedIn Post Checklist (for project write-ups)

- [ ] ROC-AUC number prominently featured
- [ ] Heatmap GIF showing anomaly region highlighted
- [ ] VRAM usage mentioned (demonstrates local reproducibility)
- [ ] JEPA / world model framing in first 2 lines
- [ ] GitHub link in first comment
