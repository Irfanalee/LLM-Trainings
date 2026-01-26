# Whiteboard AI - Prioritized Action Plan

## Overview

This document provides a step-by-step guide to get your Whiteboard Meeting Notes AI working, with priorities from "quick wins today" to "production-ready system."

---

## Phase 0: Quick Wins (TODAY - 2 hours)

### Goal: Get something working immediately without training

#### Step 1: Test the Current System (30 min)
```bash
# Create a test whiteboard image (or use a real photo)
python generate_sample.py

# Test the pipeline
python -c "
from whiteboard_ai import WhiteboardAI
ai = WhiteboardAI()

# Option A: Skip region detection (recommended for Phase 0)
results = ai.analyze_whiteboard('test_whiteboard.jpg', detect_regions=False)

# Option B: With region detection (will auto-fallback to full image if no regions found)
# results = ai.analyze_whiteboard('test_whiteboard.jpg')

print(results['full_text'])
print(results['action_items'])
"
```

> **Note:** The pretrained YOLOv8 is trained on COCO (cars, people, etc.) and won't detect whiteboard regions. The system now auto-falls back to full image OCR when no regions are detected. Use `detect_regions=False` for faster testing until you train a custom YOLOv8 model in Phase 2.

#### Step 2: Improve Prompts (30 min)
The fastest improvement with zero training:

```python
# In whiteboard_ai.py, replace the prompt in extract_action_items()
# with the improved version from scripts/improved_prompts.py

from scripts.improved_prompts import get_best_prompt, WHITEBOARD_PROMPT

# Use the whiteboard-specific prompt
prompt = get_best_prompt(meeting_notes, "whiteboard")
```

#### Step 3: Test with Real Photos (1 hour)
- Take 5-10 photos of real whiteboards
- Test OCR accuracy
- Note what works and what fails

---

## Phase 1: Generate Training Data (DAY 1)

### Goal: Create synthetic training data

#### Step 1: Generate Synthetic Whiteboards (30 min)
```bash
# Generate 500 synthetic whiteboard images with annotations
python scripts/generate_synthetic_data.py \
    --output-dir datasets/synthetic \
    --num-whiteboards 500 \
    --num-action-items 5000 \
    --num-handwriting 1000
```

#### Step 2: Verify Data Quality (15 min)
```bash
# Check generated files
ls datasets/synthetic/whiteboards/images/ | head
ls datasets/synthetic/whiteboards/labels/ | head
cat datasets/synthetic/whiteboard_yolo.yaml

# View a sample annotation
head -5 datasets/synthetic/whiteboards/labels/whiteboard_0000.txt
```

#### Step 3: Prepare Action Items for Training (15 min)
```bash
python scripts/prepare_datasets.py action \
    --input-jsonl datasets/synthetic/action_items.jsonl \
    --output-dir datasets/action_items_prepared
```

---

## Phase 2: Train YOLOv8 for Region Detection (DAY 2)

### Goal: Custom whiteboard region detection

#### Why Train YOLOv8?
- The default model detects generic objects (person, car, etc.)
- We need to detect: header, text_block, bullet_list, action_item, diagram

#### Step 1: Create Train/Val Split (5 min)
```bash
python scripts/prepare_datasets.py split \
    --dataset-dir datasets/synthetic/whiteboards \
    --val-ratio 0.2
```

#### Step 2: Train YOLOv8 (2-4 hours)
```bash
# Start with nano model (fast training)
python training/train_yolo.py \
    --data datasets/synthetic/whiteboard_yolo.yaml \
    --model n \
    --epochs 100 \
    --batch 16 \
    --output runs/whiteboard_yolo

# Monitor training
# Open another terminal and run:
# tensorboard --logdir runs/whiteboard_yolo
```

#### Step 3: Test Custom Model
```python
from ultralytics import YOLO

# Load your trained model
model = YOLO('runs/whiteboard_yolo/yolo_whiteboard_n/weights/best.pt')

# Test on a whiteboard image
results = model('test_whiteboard.jpg')
results[0].show()  # Visualize detections
```

---

## Phase 3: Fine-tune Qwen2.5 with LoRA (DAY 3-4)

### Goal: Better action item extraction

#### Why Fine-tune?
- Zero-shot works okay but misses subtle patterns
- Fine-tuning improves: JSON consistency, assignee detection, deadline parsing

#### Step 1: Train with QLoRA (3-4 hours)
```bash
python training/train_qwen_lora.py train \
    --train-data datasets/action_items_prepared/train.json \
    --val-data datasets/action_items_prepared/val.json \
    --output-dir runs/qwen_action_items \
    --epochs 3 \
    --batch-size 2 \
    --grad-accum 8
```

#### Step 2: Test the Fine-tuned Model
```bash
python training/train_qwen_lora.py test \
    --adapter runs/qwen_action_items/lora_adapter
```

#### Step 3: Integrate into Pipeline
```python
# In whiteboard_ai.py, modify the __init__ to load LoRA adapter
from peft import PeftModel

# After loading base model:
self.llm = PeftModel.from_pretrained(
    self.llm,
    "runs/qwen_action_items/lora_adapter"
)
```

---

## Phase 4: TrOCR Fine-tuning (OPTIONAL)

### When to Fine-tune TrOCR?
- Only if OCR accuracy is below 85% on your test set
- If your handwriting style is very different from training data

#### If Needed:
```bash
# Download IAM dataset (requires registration)
# https://fki.tic.heia-fr.ch/databases/iam-handwriting-database

# Prepare data
python scripts/prepare_datasets.py iam \
    --input-dir datasets/iam \
    --output-dir datasets/iam_processed

# Train with LoRA (memory efficient)
python training/train_trocr.py train \
    --train-dir datasets/iam_processed/train \
    --val-dir datasets/iam_processed/val \
    --output-dir runs/trocr_whiteboard \
    --epochs 10 \
    --batch-size 8 \
    --lora
```

---

## Phase 5: Evaluation & Iteration (DAY 5)

### Evaluate Each Component

```bash
# 1. Evaluate region detection
python evaluation/evaluate_pipeline.py yolo \
    --model runs/whiteboard_yolo/yolo_whiteboard_n/weights/best.pt \
    --images datasets/synthetic/whiteboards/val/images \
    --labels datasets/synthetic/whiteboards/val/labels

# 2. Evaluate action extraction
python evaluation/evaluate_pipeline.py action \
    --model runs/qwen_action_items/lora_adapter \
    --data datasets/action_items_prepared/val.json
```

### Target Metrics

| Component | Metric | Target | Acceptable |
|-----------|--------|--------|------------|
| YOLOv8 | mAP@0.5 | >0.80 | >0.70 |
| TrOCR | CER | <0.10 | <0.15 |
| Qwen2.5 | F1 | >0.85 | >0.75 |
| Qwen2.5 | JSON Parse Rate | >0.95 | >0.90 |

---

## Quick Reference: Commands

### Data Generation
```bash
# Generate all synthetic data
python scripts/generate_synthetic_data.py --output-dir datasets/synthetic

# Prepare for training
python scripts/prepare_datasets.py action --input-jsonl datasets/synthetic/action_items.jsonl --output-dir datasets/action_items_prepared
```

### Training
```bash
# YOLOv8 (2-4 hours)
python training/train_yolo.py --data datasets/synthetic/whiteboard_yolo.yaml --model n --epochs 100

# Qwen2.5 LoRA (3-4 hours)
python training/train_qwen_lora.py train --train-data datasets/action_items_prepared/train.json --epochs 3
```

### Evaluation
```bash
# Region detection
python evaluation/evaluate_pipeline.py yolo --model runs/whiteboard_yolo/.../best.pt --images ... --labels ...

# Action extraction
python evaluation/evaluate_pipeline.py action --model runs/qwen_action_items/lora_adapter --data ...
```

---

## Memory Requirements (RTX A4000 16GB)

| Task | VRAM Usage | Batch Size | Notes |
|------|------------|------------|-------|
| YOLOv8n training | ~4 GB | 16-32 | Fast, good for prototyping |
| YOLOv8s training | ~6 GB | 8-16 | Better accuracy |
| TrOCR full fine-tune | ~10 GB | 4-8 | Use LoRA instead |
| TrOCR LoRA | ~6 GB | 8-16 | Recommended |
| Qwen2.5-7B QLoRA | ~10-12 GB | 2 | With gradient checkpointing |

---

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
--batch-size 1 --grad-accum 16

# Enable gradient checkpointing (already in scripts)
# Clear CUDA cache
import torch
torch.cuda.empty_cache()
```

### Poor OCR Quality
1. Check image preprocessing (contrast, binarization)
2. Ensure good lighting in photos
3. Try different TrOCR models:
   - `microsoft/trocr-base-handwritten` (faster, less accurate)
   - `microsoft/trocr-large-handwritten` (current, good balance)

### Action Items Not Detected
1. Try different prompts (scripts/improved_prompts.py)
2. Lower temperature (0.1-0.2) for more consistent JSON
3. Check if meeting notes format matches training data

---

## Suggested Schedule

| Day | Task | Duration |
|-----|------|----------|
| Day 1 | Test current system, improve prompts, generate data | 4 hours |
| Day 2 | Train YOLOv8 for region detection | 4 hours |
| Day 3 | Train Qwen2.5 with LoRA | 4 hours |
| Day 4 | Integration and testing | 4 hours |
| Day 5 | Evaluation, iteration, demo prep | 4 hours |

**Total: ~20 hours to production-ready system**

---

## Files Created

```
whiteboard-ai/
├── training/
│   ├── ACTION_PLAN.md          # This file
│   ├── DATASETS_GUIDE.md       # Dataset links and instructions
│   ├── train_yolo.py           # YOLOv8 training script
│   ├── train_trocr.py          # TrOCR training script
│   └── train_qwen_lora.py      # Qwen2.5 LoRA training script
├── scripts/
│   ├── generate_synthetic_data.py  # Synthetic data generation
│   ├── prepare_datasets.py         # Data preparation utilities
│   └── improved_prompts.py         # Better prompts for zero-shot
├── evaluation/
│   └── evaluate_pipeline.py    # Evaluation scripts
└── datasets/                   # Generated data goes here
```
