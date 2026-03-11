# Runbooks — WorldGuard Operational Procedures

## Runbook 1: New Camera Onboarding

**Goal**: Get a new camera producing calibrated anomaly scores.

**Steps**:
1. Record at least 2 hours of normal footage from the new camera
2. Extract clips:
   ```bash
   python data/extract_clips.py \
     --video-dir data/raw/cam05/ \
     --output-dir data/train/cam05/ \
     --stride 2 --num-frames 16
   ```
3. Run threshold calibration using pretrained model:
   ```bash
   python training/calibrate.py \
     --checkpoint checkpoints/best.pt \
     --val-dir data/train/cam05/ \
     --camera-id cam05 \
     --output configs/thresholds/cam05.json
   ```
4. Verify with a test clip:
   ```bash
   python inference/score_video.py \
     --video data/test_normal/cam05_normal.mp4 \
     --camera-id cam05
   ```
5. Expected: anomaly score < threshold. If not, re-run calibration with more clips.

---

## Runbook 2: Retraining on New Normal Data

**When to use**: Model false positive rate is too high after scene change (lighting, camera moved).

**Steps**:
1. Collect fresh normal footage (minimum 4 hours)
2. Update `configs/train_default.yaml` — set `finetune_from: checkpoints/best.pt`
3. Run finetuning (fewer epochs since we're adapting, not training from scratch):
   ```bash
   python training/train.py \
     --config configs/finetune_cam05.yaml \
     --epochs 20
   ```
4. Re-run threshold calibration for affected cameras
5. Validate on held-out clips before deploying

---

## Runbook 3: Debugging High False Positive Rate

See `.claude/skills/debugging-flow.md` → Section "Anomaly Threshold Issues"

Quick checks:
- Is the camera position/angle unchanged?
- Is there a new regular pattern in the scene (e.g. construction started)?
- Run `python eval/visualize_errors.py` to see score distribution shift

---

## Runbook 4: Evaluating on UCSD Ped2 Benchmark

```bash
# Download dataset
# http://www.svcl.ucsd.edu/projects/anomaly/dataset.html
# Place in data/ucsd_ped2/

# Run evaluation
python eval/eval_roc.py \
  --checkpoint checkpoints/best.pt \
  --test-dir data/ucsd_ped2/ \
  --output outputs/ucsd_ped2_results.json

# Results will print: Frame-level AUROC, ROC curve PNG
```

Target: AUROC > 0.80 on Ped2 (competitive with published unsupervised methods).
