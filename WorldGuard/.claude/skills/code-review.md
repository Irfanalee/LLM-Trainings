# Skill: Code Review Checklist

Use this checklist whenever reviewing or writing code in WorldGuard.

## General
- [ ] No hardcoded paths — all paths come from config or argparse
- [ ] Functions have a single clear responsibility
- [ ] Variable names reflect what the tensor/data actually is (e.g. `z_ctx` not `x1`)
- [ ] No silent failures — exceptions are raised or logged explicitly

## PyTorch Specific
- [ ] `torch.no_grad()` is used anywhere the target encoder is called
- [ ] `model.eval()` is set before inference; `model.train()` before training
- [ ] Loss `.item()` is called before logging to avoid accumulating computation graph
- [ ] Tensors are explicitly moved to the right device — no assumptions about CPU/GPU
- [ ] No in-place ops on tensors that require grad (`.add_` etc. — causes autograd issues)
- [ ] `DataLoader` uses `num_workers > 0` and `pin_memory=True` for GPU training

## JEPA-Specific Rules
- [ ] Target encoder parameters are NEVER in the optimizer
- [ ] EMA update (`update_target_encoder()`) is called AFTER optimizer step, not before
- [ ] Masking is applied to FUTURE (target) patches only — context frames are always visible
- [ ] Loss is computed in latent space — never pixel MSE
- [ ] Batch size + sequence length checked against 16GB VRAM budget before merging

## Video Data
- [ ] Clips are loaded via PyAV, not OpenCV VideoCapture
- [ ] Frame extraction uses consistent stride (default: 2) — not random skip
- [ ] Augmentations are applied consistently across all frames in a clip (same crop, same flip)
- [ ] No data leakage — val/test clips come from different recording sessions than train

## Anomaly Scoring
- [ ] Threshold is loaded per camera from `configs/thresholds/{camera_id}.json`
- [ ] Score and heatmap are both returned — never just the binary alert
- [ ] Heatmap is normalized to [0, 1] before visualization

## Before Merging Any Training Code
- [ ] Loss curve is decreasing and not exploding in first 100 steps
- [ ] W&B run is logging: loss, learning_rate, EMA decay, GPU memory
- [ ] Config file is committed alongside the code change
