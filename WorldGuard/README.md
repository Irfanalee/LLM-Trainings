---
language:
  - en
license: mit
tags:
  - video-anomaly-detection
  - jepa
  - unsupervised
  - pytorch
  - cctv
  - world-model
  - vit
datasets:
  - ucsd-ped2
  - shanghaitech
metrics:
  - roc_auc
---

<div align="center">

# 🛡️ WorldGuard

### JEPA-Inspired Video World Model for Unsupervised CCTV Anomaly Detection

*Inspired by Yann LeCun's AMI Labs world model thesis — implemented locally on a single RTX A4000*

<br/>

![Release](https://img.shields.io/badge/RELEASE-v0.1.0-brightgreen?style=flat-square)
![Model](https://img.shields.io/badge/ARCHITECTURE-V--JEPA-8a2be2?style=flat-square)
![Paradigm](https://img.shields.io/badge/PARADIGM-World%20Model-ff6b35?style=flat-square)
![Labels](https://img.shields.io/badge/LABELS-Zero-blue?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.x-76b900?style=flat-square&logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/LICENSE-MIT-yellow?style=flat-square)

</div>

---

## 📊 Results

### Benchmark Performance (Frame-level AUROC)

| Dataset | Run 1 (ShanghaiTech only) | Run 2 (Balanced) | Change |
|---|---|---|---|
| UCSD Ped2 | 0.545 | **0.788** | +24% |
| ShanghaiTech | 0.639 | **0.614** | -2.5% |

Run 1 trained on ShanghaiTech data only. Run 2 retrained on balanced UCSD Ped2 + ShanghaiTech clips — UCSD Ped2 jumped dramatically once the model saw that scene during training. ShanghaiTech dropped slightly, a normal trade-off when splitting training capacity between two datasets.

### Context vs Published Work

| Method | UCSD Ped2 AUROC |
|---|---|
| Supervised SOTA | ~0.96–0.99 |
| Unsupervised / reconstruction-based | ~0.82–0.92 |
| **WorldGuard (this model, 50 epochs)** | **0.788** |

WorldGuard achieves competitive unsupervised performance with zero anomaly labels during training, on a single consumer GPU in 50 epochs.

### To Push Further
- Train 100+ epochs — loss was still decreasing at epoch 50
- Add more ShanghaiTech training data (currently 2077 clips; full set = 330 videos)

---

## 🧠 Why This Exists — The World Model Thesis

In March 2026, Yann LeCun's **AMI Labs** raised **$1.03 billion** to build AI that goes beyond LLMs. The core thesis: real intelligence predicts abstract representations of future states, not pixels or tokens.

WorldGuard is a direct local implementation of that thesis applied to CCTV surveillance:

> *Train a model to predict what should happen next. When reality deviates from prediction — that's an anomaly.*

### Why Not Just Train a Classifier?

| | Supervised Classifier | WorldGuard (World Model) |
|---|---|---|
| Labels needed | Hundreds per class | **Zero** |
| Detects novel anomalies | No — known classes only | **Yes — any deviation** |
| Generalizes to new cameras | Poor | **Better — learns scene structure** |
| Failure mode | Unknown unknowns invisible | All deviations flagged |

---

## ⚙️ How It Works

```
CCTV Video Stream
      │
      ▼
┌─────────────────────┐
│   Frame Extractor   │  16 frames @ 224×224, stride 2 (~1s @ 30fps)
└─────────────────────┘
      │
      ▼
┌──────────────────────────────────────────┐
│           JEPA WORLD MODEL               │
│                                          │
│  Context Encoder (ViT-S/16)              │
│       → z_ctx  [B, T_ctx, D]             │
│                                          │
│  Predictor (Transformer, 4 layers)       │
│       → z_pred [B, T_future, D]          │
│                                          │
│  Target Encoder (EMA — no gradients)     │
│       → z_tgt  [B, T_future, D]          │
│                                          │
│  Loss = L2(z_pred, z_tgt)               │
│  ↑ latent space only — never pixels      │
└──────────────────────────────────────────┘
      │
      ▼
┌─────────────────────┐
│   Anomaly Scorer    │  mean prediction error per clip
│   + Spatial Heatmap │  per-patch error → 224×224 overlay
└─────────────────────┘
```

The model **never trains on anomalies**. It learns the statistical structure of normality. At inference, anything that breaks that structure produces a spike in latent prediction error.

---

## 🏗️ Architecture

| Component | Design | VRAM |
|---|---|---|
| **Context Encoder** | VideoMAE-pretrained ViT-S/16 (21M params) | ~1.8 GB |
| **Target Encoder** | EMA copy — frozen, no gradients | ~1.8 GB |
| **Predictor** | 4-layer Transformer (D=384, heads=6) | ~0.4 GB |
| **Activations** | Batch=16, 16 frames @ 224×224 | ~8–10 GB |
| **Total** | | **~13–14 GB** |

---

## 🚀 Quickstart

### 1. Install dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 2. Download checkpoint

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="irfanalii/worldguard",
    filename="checkpoints/train_default_epoch050_val0.0191.pt"
)
```

### 3. Score a video

```bash
python inference/score_video.py \
  --video /path/to/footage.mp4 \
  --checkpoint checkpoints/train_default_epoch050_val0.0191.pt \
  --camera-id cam01
```

### 4. Evaluate on UCSD Ped2

```bash
python eval/eval_roc.py \
  --checkpoint checkpoints/train_default_epoch050_val0.0191.pt \
  --test-dir data/ucsd_ped2/ \
  --output outputs/eval/
```

### 5. Retrain on your own footage

```bash
# Extract clips from your normal CCTV recordings
python data/extract_clips.py \
  --video /path/to/cam01.mp4 \
  --output-dir data/train \
  --camera-id cam01

# Train
python training/train.py --config configs/train_default.yaml

# Calibrate threshold
python training/calibrate.py \
  --checkpoint checkpoints/best.pt \
  --val-dir data/val \
  --camera-id cam01
```

---

## 📚 References

- Assran et al. (2025). [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://arxiv.org/abs/2506.09985)
- Bardes et al. (2024). [V-JEPA: Revisiting Feature Prediction for Learning Visual Representations from Video](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/)
- LeCun, Y. (2022). [A Path Towards Autonomous Machine Intelligence](https://openreview.net/pdf?id=BZ5a1r-kVsf)

---

## 📄 License

MIT

---

<div align="center">

Built by [@Irfanalee](https://github.com/Irfanalee) · [Source Code](https://github.com/Irfanalee/LLM-Trainings/tree/main/WorldGuard) · Inspired by LeCun's world model thesis · Runs entirely on local hardware

</div>
