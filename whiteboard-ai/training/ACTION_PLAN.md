# Whiteboard AI - Prioritized Action Plan

## Overview

This document provides a step-by-step guide to fine-tune models for the Whiteboard Meeting Notes AI using **real datasets**.

**Datasets Downloaded:**
- **Whiteboard Detection** (Roboflow): 713 images, YOLO format
- **IAM Handwriting**: Real handwritten words with transcriptions

---

## Phase 0: Quick Wins (COMPLETED)

### Summary
- Tested pipeline with pretrained models
- Identified issues: TrOCR needs line-level images, YOLOv8 needs whiteboard training
- See [step0.md](step0.md) for details

---

## Phase 1: Verify Downloaded Datasets

### Goal: Ensure datasets are ready for training

#### Step 1: Check Whiteboard Detection Dataset
```bash
# Verify structure
ls "datasets/WhiteBoard Detection.v1i.yolov8/"
# Expected: train/, valid/, test/, data.yaml

# Check data.yaml
cat "datasets/WhiteBoard Detection.v1i.yolov8/data.yaml"
# Expected: nc: 3, names: ['WhiteBoard', 'board', 'people']

# Count images
ls "datasets/WhiteBoard Detection.v1i.yolov8/train/images/" | wc -l
# Expected: ~713 images
```

#### Step 2: Check IAM Handwriting Dataset
```bash
# Verify structure
ls datasets/Handwriting/
# Expected: iam_words/, words_new.txt

# Check transcription format
head -10 datasets/Handwriting/words_new.txt
# Format: image_id status graylevel components x y w h tag transcription

# Check sample images exist
ls datasets/Handwriting/iam_words/words/a01/ | head -5
```

---

## Phase 2: Train YOLOv8 for Whiteboard Detection

### Goal: Detect whiteboards in photos

#### What You'll Learn
- How to use a pretrained model as starting point
- YOLO training loop and metrics (mAP, precision, recall)
- Transfer learning: COCO weights → whiteboard detection

#### Dataset Info
| Property | Value |
|----------|-------|
| Training images | 713 |
| Classes | WhiteBoard, board, people |
| Format | YOLO (ready to use) |

#### Step 1: Train YOLOv8 (1-2 hours)
```bash
cd /home/irfana/Documents/repos/LLM-Trainings/whiteboard-ai

python training/train_yolo.py \
    --data "datasets/WhiteBoard Detection.v1i.yolov8/data.yaml" \
    --model n \
    --epochs 50 \
    --batch 16 \
    --output runs/whiteboard_yolo
```

**What each argument does:**
- `--data`: Path to dataset config (tells YOLO where images/labels are)
- `--model n`: YOLOv8-nano (smallest, ~6MB, fastest training)
- `--epochs 50`: 50 passes through all training data
- `--batch 16`: Process 16 images at a time (fits in 16GB VRAM)
- `--output`: Where to save trained model

#### Step 2: Monitor Training
```bash
# In another terminal
tensorboard --logdir runs/whiteboard_yolo
# Open http://localhost:6006
```

#### Step 3: Test Your Trained Model
```python
from ultralytics import YOLO

# Load your trained model
model = YOLO('runs/whiteboard_yolo/yolo_whiteboard_n/weights/best.pt')

# Test on an image
results = model('path/to/whiteboard_photo.jpg')
results[0].show()  # Visualize detections
```

#### Expected Results
| Metric | Target | Notes |
|--------|--------|-------|
| mAP@0.5 | >0.70 | Mean Average Precision |
| Precision | >0.80 | Correct detections / All detections |
| Recall | >0.70 | Correct detections / All actual objects |

---

## Phase 3: Fine-tune TrOCR on IAM Handwriting

### Goal: Improve handwriting recognition

#### What You'll Learn
- How to prepare image-text pairs for OCR training
- Fine-tuning a Vision-Encoder-Decoder model
- Character Error Rate (CER) as evaluation metric

#### Dataset Info
| Property | Value |
|----------|-------|
| Source | IAM Handwriting Database |
| Content | Individual handwritten words |
| Format | Images + transcriptions in words_new.txt |

#### Step 1: Prepare IAM Data
```bash
python scripts/prepare_datasets.py iam \
    --input-dir datasets/Handwriting \
    --output-dir datasets/iam_processed
```

This will:
- Parse words_new.txt to get image paths and transcriptions
- Split into train/val sets (80/20)
- Create JSON metadata files for training

#### Step 2: Train TrOCR (2-4 hours)
```bash
python training/train_trocr.py train \
    --train-dir datasets/iam_processed/train \
    --val-dir datasets/iam_processed/val \
    --output-dir runs/trocr_handwriting \
    --epochs 5 \
    --batch-size 8 \
    --lora
```

**What each argument does:**
- `--train-dir`: Folder with training images + metadata
- `--val-dir`: Folder with validation images + metadata
- `--lora`: Use LoRA (Low-Rank Adaptation) to save memory
- `--epochs 5`: 5 passes through data (OCR converges fast)
- `--batch-size 8`: 8 images at a time

#### Step 3: Test Your Trained TrOCR
```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from peft import PeftModel
from PIL import Image

# Load base model + your LoRA adapter
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
model = PeftModel.from_pretrained(model, "runs/trocr_handwriting/lora_adapter")

# Test on an image
image = Image.open("path/to/handwriting.png").convert("RGB")
pixel_values = processor(images=image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)
text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(text)
```

#### Expected Results
| Metric | Target | Notes |
|--------|--------|-------|
| CER | <0.15 | Character Error Rate (lower is better) |
| WER | <0.30 | Word Error Rate |

---

## Phase 4: Fine-tune Qwen2.5 with LoRA

### Goal: Better action item extraction from meeting notes

#### What You'll Learn
- QLoRA: 4-bit quantization + LoRA for memory efficiency
- Training LLMs on instruction-following tasks
- JSON output formatting

#### Step 1: Generate Synthetic Action Items Data
```bash
# We need meeting notes → action items pairs
# Generate synthetic data since we don't have real meeting notes
python scripts/generate_synthetic_data.py \
    --output-dir datasets/synthetic \
    --num-action-items 5000 \
    --num-whiteboards 0 \
    --num-handwriting 0
```

#### Step 2: Prepare for Training
```bash
python scripts/prepare_datasets.py action \
    --input-jsonl datasets/synthetic/action_items.jsonl \
    --output-dir datasets/action_items_prepared
```

#### Step 3: Train Qwen2.5 with QLoRA (3-4 hours)
```bash
python training/train_qwen_lora.py train \
    --train-data datasets/action_items_prepared/train.json \
    --val-data datasets/action_items_prepared/val.json \
    --output-dir runs/qwen_action_items \
    --epochs 3 \
    --batch-size 2 \
    --grad-accum 8
```

**What each argument does:**
- `--batch-size 2`: Small batch (LLMs are memory hungry)
- `--grad-accum 8`: Accumulate gradients over 8 steps = effective batch of 16
- `--epochs 3`: LLMs need fewer epochs (they learn fast)

#### Step 4: Test Your Fine-tuned Model
```bash
python training/train_qwen_lora.py test \
    --adapter runs/qwen_action_items/lora_adapter
```

#### Step 5: Integrate into Pipeline
```python
# In whiteboard_ai.py, modify __init__ to load LoRA adapter
from peft import PeftModel

# After loading base model:
self.llm = PeftModel.from_pretrained(
    self.llm,
    "runs/qwen_action_items/lora_adapter"
)
```

#### Expected Results
| Metric | Target | Notes |
|--------|--------|-------|
| F1 Score | >0.80 | Precision-Recall balance |
| JSON Parse Rate | >0.95 | Valid JSON output |

---

## Phase 5: Integration & Evaluation

### Goal: Put it all together and measure performance

#### Step 1: Update whiteboard_ai.py
```python
# Load your trained models instead of pretrained
self.region_model = YOLO('runs/whiteboard_yolo/.../best.pt')
self.ocr_model = PeftModel.from_pretrained(base_ocr, 'runs/trocr_handwriting/lora_adapter')
self.llm = PeftModel.from_pretrained(base_llm, 'runs/qwen_action_items/lora_adapter')
```

#### Step 2: Run Full Pipeline Evaluation
```bash
python evaluation/evaluate_pipeline.py full \
    --yolo-model runs/whiteboard_yolo/.../best.pt \
    --trocr-adapter runs/trocr_handwriting/lora_adapter \
    --qwen-adapter runs/qwen_action_items/lora_adapter \
    --test-images datasets/test_whiteboards/
```

#### Step 3: Test with Real Photos
- Take photos of real whiteboards with your phone
- Run through the pipeline
- Check if action items are correctly extracted

---

## Quick Reference: Commands

### Training Commands
```bash
# YOLOv8 (1-2 hours)
python training/train_yolo.py \
    --data "datasets/WhiteBoard Detection.v1i.yolov8/data.yaml" \
    --model n --epochs 50 --batch 16

# TrOCR (2-4 hours)
python training/train_trocr.py train \
    --train-dir datasets/iam_processed/train \
    --val-dir datasets/iam_processed/val \
    --epochs 5 --batch-size 8 --lora

# Qwen2.5 LoRA (3-4 hours)
python training/train_qwen_lora.py train \
    --train-data datasets/action_items_prepared/train.json \
    --epochs 3 --batch-size 2 --grad-accum 8
```

---

## Memory Requirements (RTX A4000 16GB)

| Task | VRAM Usage | Batch Size | Notes |
|------|------------|------------|-------|
| YOLOv8n training | ~4 GB | 16-32 | Fast, good for learning |
| TrOCR LoRA | ~6 GB | 8-16 | Memory efficient |
| Qwen2.5-7B QLoRA | ~10-12 GB | 2 | With gradient checkpointing |

---

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
--batch-size 1 --grad-accum 16

# Clear CUDA cache
python -c "import torch; torch.cuda.empty_cache()"
```

### Poor Training Results
1. Check if data paths are correct
2. Verify labels match images
3. Try more epochs or lower learning rate

---

## Schedule Summary

| Phase | Task | Time |
|-------|------|------|
| 0 | Test pretrained models | Done |
| 1 | Verify datasets | 15 min |
| 2 | Train YOLOv8 | 1-2 hours |
| 3 | Fine-tune TrOCR | 2-4 hours |
| 4 | Fine-tune Qwen2.5 | 3-4 hours |
| 5 | Integration & testing | 2 hours |

**Total: ~10-12 hours**

---

## Files Structure

```
whiteboard-ai/
├── training/
│   ├── ACTION_PLAN.md          # This file
│   ├── step0.md                # Phase 0 results
│   ├── train_yolo.py           # YOLOv8 training script
│   ├── train_trocr.py          # TrOCR training script
│   └── train_qwen_lora.py      # Qwen2.5 LoRA training script
├── datasets/
│   ├── WhiteBoard Detection.v1i.yolov8/  # Roboflow dataset (713 images)
│   ├── Handwriting/                       # IAM dataset
│   └── synthetic/                         # Generated action items
├── runs/                       # Training outputs go here
│   ├── whiteboard_yolo/
│   ├── trocr_handwriting/
│   └── qwen_action_items/
└── evaluation/
    └── evaluate_pipeline.py
```
