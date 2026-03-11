# Skill: Debugging Flow

Follow this process when something breaks in WorldGuard. Do not jump to fixes before diagnosing.

## Step 1 — Identify the failure category

| Symptom | Category |
|---|---|
| Loss is NaN from step 1 | Initialization or data issue |
| Loss explodes after N steps | LR too high or gradient issue |
| Loss plateau / not decreasing | EMA not updating, predictor too weak |
| OOM on GPU | Batch size, sequence length, or gradient accumulation issue |
| Anomaly score never exceeds threshold | Threshold too high or model undertrained |
| Anomaly score always exceeds threshold | Threshold too low or distribution shift |
| Heatmap is uniform / all same color | Score aggregation bug or encoder collapse |

## Step 2 — Check the basics first (in order)

1. **Check GPU memory**: `nvidia-smi` — is VRAM actually available?
2. **Check data loading**: print one batch shape, min/max pixel values, dtype
3. **Check model forward pass in isolation**: run with dummy data `torch.randn(...)` before real data
4. **Check gradients**: after first backward, `print(loss.item())` — is it finite?
5. **Check EMA encoder**: confirm `target_encoder` params are NOT in `optimizer.param_groups`

## Step 3 — NaN Loss Debugging

```python
# Add after loss computation
if torch.isnan(loss):
    print("z_ctx stats:", z_ctx.min(), z_ctx.max(), z_ctx.mean())
    print("z_pred stats:", z_pred.min(), z_pred.max())
    print("z_tgt stats:", z_tgt.min(), z_tgt.max())
    raise ValueError("NaN loss detected — check above stats")
```

Common causes:
- Learning rate too high (try 1e-5 first, then scale up)
- Missing LayerNorm in predictor
- Encoder produces zero vectors (collapse) — add variance regularization

## Step 4 — Encoder Collapse Check

Collapse = encoder maps all inputs to the same embedding. Signs:
- Loss decreases to near-zero too fast (< 10 steps)
- t-SNE shows single cluster for all clips

Fix:
- Confirm EMA decay is set correctly (should be 0.996, not 0.0)
- Confirm `stop_gradient` / `torch.no_grad()` is on target encoder
- Add VICReg or variance regularization if collapse persists

## Step 5 — OOM Debugging

In order of impact:
1. Reduce `batch_size` in config (try 8, then 4)
2. Enable gradient checkpointing: `encoder.gradient_checkpointing_enable()`
3. Reduce `num_frames` from 16 to 8
4. Reduce `image_size` from 224 to 160
5. Use `torch.cuda.amp` (mixed precision) — should always be ON anyway

## Step 6 — Anomaly Threshold Issues

If threshold seems wrong after calibration:
- Re-run `calibrate.py` with more validation clips (min 100 clips recommended)
- Plot the score distribution as a histogram — is it bimodal or unimodal?
- Check for scene changes in val data (day→night transitions inflate std dev)
- Try `mean + 3.0 * std` instead of `2.5` for lower false positive rate

## Useful Debug Commands

```bash
# Check GPU
nvidia-smi

# Profile a training step
python -c "import torch; print(torch.cuda.memory_summary())"

# Quick forward pass test
python -c "
from models.jepa_model import JEPAWorldModel
import torch
model = JEPAWorldModel(...).cuda()
x = torch.randn(2, 16, 3, 224, 224).cuda()
loss, _, _ = model(x[:, :12], x[:, 12:], mask=None)
print('Loss:', loss.item())
"
```
