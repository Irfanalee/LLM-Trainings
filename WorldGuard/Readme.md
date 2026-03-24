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

![Hardware](https://img.shields.io/badge/Hardware-RTX%20A4000%2016GB-76b900?style=flat-square&logo=nvidia&logoColor=white)
![VRAM](https://img.shields.io/badge/VRAM-~13--14GB-orange?style=flat-square)
![Self--Supervised](https://img.shields.io/badge/Training-Self--Supervised-success?style=flat-square)
![GitHub stars](https://img.shields.io/github/stars/irfanalii/WorldGuard?style=flat-square&logo=github)

</div>

---

## 🧠 Why This Exists — The World Model Thesis

In March 2026, Yann LeCun's **AMI Labs** raised **$1.03 billion** — Europe's largest seed round ever — to build AI that goes beyond LLMs. The core thesis: real intelligence doesn't start in language. It starts in the world.

LeCun's framework, **JEPA (Joint Embedding Predictive Architecture)**, predicts abstract representations of future states rather than generating pixels or tokens. The model learns the structure of reality — not a catalog of known events.

WorldGuard is a direct local implementation of that thesis applied to CCTV surveillance:

> *Train a model to predict what should happen next. When reality deviates from prediction — that's an anomaly.*

### Why Not Just Train a Classifier?

Standard supervised anomaly detection has a fatal flaw: **you can only detect what you've seen before.** A fire detector trained on flames won't catch smoke-only fires. A fighting detector trained on punches won't catch a quiet threatening confrontation.

| | Supervised Classifier | WorldGuard (World Model) |
|---|---|---|
| Labels needed | Hundreds per class | **Zero** |
| Detects novel anomalies | No — known classes only | **Yes — any deviation** |
| Generalizes to new cameras | Poor | **Better — learns scene structure** |
| Failure mode | Unknown unknowns invisible | All deviations flagged |
| Inspired by | ImageNet-era supervised AI | LeCun's JEPA / AMI Labs |

---

## ⚙️ How It Works

WorldGuard implements a **JEPA-style video world model** in four stages:

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
      │
      ▼
  Alert  │  Score  │  Annotated Video
```

### The Key Insight

We **never train on anomalies at all.** The model trains only on normal footage. The JEPA predictor learns the statistical structure of normality. At inference, anything that breaks that structure — fire, intrusion, someone running where people normally walk — produces a **spike in latent prediction error**.

This is closer to how humans notice anomalies: not by recognizing known threats, but by detecting that something doesn't fit the expected pattern.

---

## 🏗️ Architecture

### Model Components

| Component | Design | VRAM |
|---|---|---|
| **Context Encoder** | VideoMAE-pretrained ViT-S/16 (21M params) | ~1.8 GB |
| **Target Encoder** | EMA copy of context encoder — frozen, no gradients | ~1.8 GB |
| **Predictor** | 4-layer Transformer (D=384, heads=6) | ~0.4 GB |
| **Activations** | Batch=16, 16 frames @ 224×224 | ~8–10 GB |
| **Optimizer states** | AdamW | ~1.4 GB |
| **Total** | Fits RTX A4000 with margin | **~13–14 GB** |

### Why ViT-S/16?

ViT-L (307M params) would require ~22GB VRAM. ViT-S/16 (21M params) fits comfortably at batch 16, trains 8× faster, and produces representations rich enough for anomaly scoring — not fine-grained classification. The difference in AUROC is marginal for this task.

### Why EMA Target Encoder?

EMA prevents representation collapse without needing negative pairs (unlike contrastive methods). The target encoder slowly tracks the context encoder → stable training targets. Same mechanism as V-JEPA, BYOL, and DINO.

---

## 📁 Repository Structure

```
WorldGuard/
├── CLAUDE.md                       # AI briefing doc — Claude Code reads this first
├── .claude/
│   ├── settings.json               # Hooks: ruff autofix after every edit
│   └── skills/
│       ├── code-review.md          # PyTorch + JEPA-specific review checklist
│       ├── debugging-flow.md       # NaN loss, collapse, OOM, threshold debugging
│       └── release-procedure.md    # Experiment + benchmark + LinkedIn checklist
│
├── configs/
│   ├── train_default.yaml          # Hyperparameters — all paths and settings here
│   └── thresholds/                 # Per-camera threshold JSONs (cam01.json, etc.)
│
├── data/
│   ├── extract_clips.py            # PyAV clip extractor — run this first
│   ├── dataset.py                  # ClipDataset → (context_frames, target_frames)
│   └── augmentations.py            # ConsistentAugment — same crop/flip across all frames
│
├── models/
│   ├── CLAUDE.md                   # "EMA encoder must NEVER receive gradients"
│   ├── encoder.py                  # ViT-S/16 context + EMA target encoder
│   ├── predictor.py                # Temporal predictor transformer
│   └── jepa_model.py               # Full JEPA forward pass + EMA update logic
│
├── training/
│   ├── CLAUDE.md                   # "Labels are forbidden in train.py"
│   ├── train.py                    # Main training loop — AdamW + cosine LR + W&B
│   ├── calibrate.py                # Threshold calibration: mean + 2.5σ per camera
│   └── utils.py                    # EMA update, logging, checkpointing
│
├── inference/
│   ├── CLAUDE.md                   # Heatmap contracts, threshold loading rules
│   ├── score_video.py              # Run anomaly scoring on new footage
│   ├── heatmap.py                  # Per-patch error → 14×14 → upsample to 224×224
│   └── demo.py                     # Live webcam / RTSP stream demo
│
├── eval/
│   ├── eval_roc.py                 # Frame-level AUROC on UCSD Ped2
│   └── visualize_errors.py         # t-SNE: normal vs anomaly embedding clusters
│
├── docs/
│   ├── architecture.md             # System design decisions
│   ├── runbooks.md                 # Camera onboarding, retraining, eval procedures
│   └── team-decisions.md           # ADRs and benchmark results log
│
├── checkpoints/                    # Saved weights — gitignored
└── outputs/                        # Anomaly clips, heatmaps — gitignored
```

---

## 🤖 Claude Code Setup

This repo is built to work with **Claude Code** (VS Code). The `.claude/` directory contains skills and hooks that make Claude a reliable coding partner on this project.

### How it's structured

The project follows a **layered context** approach — Claude reads the minimum needed for any given task:

```
CLAUDE.md           ← Always read first. What, Why, Where, Rules, State.
    │
    ├── docs/       ← Progressive context. Claude reads these when needed.
    │               └── architecture.md, runbooks.md, team-decisions.md
    │
    └── src/*/      ← Local CLAUDE.md in risky directories.
        CLAUDE.md       models/ → EMA gradient rule
                        training/ → no-labels rule
                        inference/ → heatmap contract
```

### Skills (`.claude/skills/`)

Playbooks Claude uses consistently across sessions:

- **`code-review.md`** — PyTorch-specific checklist: EMA order, no-grad checks, VRAM budget, data leakage guards
- **`debugging-flow.md`** — Systematic flow for NaN loss, encoder collapse, OOM, threshold issues
- **`release-procedure.md`** — Experiment logging, checkpoint validation, LinkedIn post checklist

### Hooks (`.claude/settings.json`)

Automated actions that fire on every file edit — models never forget, hooks never do:

- **`ruff --fix`** runs after every Python file edit — auto-lints silently
- **Syntax validation** on every `models/` change — catches broken imports immediately

---

## 🚀 Quickstart

### 1. Install dependencies

```bash
conda create -n worldguard python=3.10
conda activate worldguard
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install timm einops av wandb ruff scikit-learn matplotlib opencv-python
```

### 2. Prepare training data (normal footage only — no labels needed)

```bash
# Extract 16-frame clips from your CCTV recordings
python data/extract_clips.py \
  --video-dir data/raw/cam01/ \
  --output-dir data/train/cam01/ \
  --stride 2 --num-frames 16
```

> **Don't have CCTV footage yet?** Download [UCSD Ped2](http://www.svcl.ucsd.edu/projects/anomaly/dataset.html) — use the `Train/` folder (normal only) to get started.

### 3. Train the world model

```bash
python training/train.py --config configs/train_default.yaml
```

### 4. Calibrate per-camera thresholds

```bash
python training/calibrate.py \
  --checkpoint checkpoints/best.pt \
  --val-dir data/val/cam01/ \
  --camera-id cam01
```

### 5. Score new footage

```bash
python inference/score_video.py \
  --video data/test/suspicious_clip.mp4 \
  --camera-id cam01
```

### 6. Evaluate on UCSD Ped2

```bash
python eval/eval_roc.py \
  --checkpoint checkpoints/best.pt \
  --test-dir data/ucsd_ped2/ \
  --output outputs/eval/
```

---

## 📊 Training Data Sources

| Dataset | Use | Download |
|---|---|---|
| **UCSD Ped2** | Primary benchmark — overhead pedestrian CCTV | [svcl.ucsd.edu](http://www.svcl.ucsd.edu/projects/anomaly/dataset.html) |
| **CUHK Avenue** | Campus walkway, running/throwing anomalies | [cse.cuhk.edu.hk](http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal) |
| **ShanghaiTech** | 13 scenes, 130 anomaly clips | [GitHub](https://github.com/StevenLiuWen/sRNN_TSC_Anomaly_Detection) |
| **UCF-Crime** | Real CCTV, 13 crime categories | [paperswithcode](https://paperswithcode.com/dataset/ucf-crime) |
| **Your own camera** | Best for deployment — record 2–4hrs of normal footage | Any webcam / IP cam |

Training uses the **normal splits only**. Anomaly clips are reserved exclusively for evaluation.

---

## 📚 References

- Assran et al. (2025). [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://arxiv.org/abs/2506.09985). *arXiv:2506.09985*
- Bardes et al. (2024). [V-JEPA: Revisiting Feature Prediction for Learning Visual Representations from Video](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/). *Meta AI Research*
- LeCun, Y. (2022). [A Path Towards Autonomous Machine Intelligence](https://openreview.net/pdf?id=BZ5a1r-kVsf). *Meta AI Research*
- AMI Labs / Advanced Machine Intelligence. [$1.03B Seed Round Announcement](https://capacityglobal.com/news/yann-lecun-ami-labs-raises-1bn-world-models-ai-funding/). *March 2026*

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built by [@irfanalii](https://github.com/irfanalii) · Inspired by LeCun's world model thesis · Runs entirely on local hardware

</div>