# CLAUDE.md — infra/

## Guardrails

This folder contains any deployment or environment setup scripts.

### No Cloud Dependency
WorldGuard is designed to run 100% locally on the RTX A4000. Do not add:
- Cloud storage (S3, GCS) as a hard dependency
- Docker images requiring internet at runtime
- Any paid API calls

### Environment Setup
```bash
# Reproducible environment
conda create -n worldguard python=3.10
conda activate worldguard
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install timm einops opencv-python av wandb ruff
```

### CUDA Version
RTX A4000 runs CUDA 12.x. Always pin to `torch+cu121` or `torch+cu124`.
Do NOT use CPU-only torch builds — training will be ~100x slower.

### gitignore — These Must Never Be Committed
```
checkpoints/
outputs/
data/raw/
data/train/
data/val/
*.mp4
*.avi
*.pt
*.pth
wandb/
```
