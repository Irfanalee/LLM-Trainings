# Whiteboard AI Pipeline

## Overview

```
Input Image → Region Detection → OCR → Action Extraction → Structured Output
                (YOLOv8)       (TrOCR/EasyOCR)  (Qwen2.5)
```

---

## Pipeline Stages

### Stage 1: Region Detection
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Whiteboard │ --> │   YOLOv8    │ --> │  Bounding   │
│    Image    │     │  (trained)  │     │    Boxes    │
└─────────────┘     └─────────────┘     └─────────────┘
```
- **Model**: YOLOv8-nano (fine-tuned)
- **Dataset**: Roboflow Whiteboard Detection
- **Classes**: WhiteBoard, board, people

---

### Stage 2: OCR (Text Extraction)

```
┌─────────────────────────────────────────────────────┐
│                   OCR Options                       │
├─────────────┬─────────────────┬─────────────────────┤
│  EasyOCR    │     Hybrid      │    TrOCR Only       │
│  (default)  │   (recommended) │   (single-line)     │
├─────────────┼─────────────────┼─────────────────────┤
│  CRAFT +    │  EasyOCR detect │  TrOCR + LoRA       │
│  Recognition│  + TrOCR read   │  (needs pre-crop)   │
└─────────────┴─────────────────┴─────────────────────┘
```

- **TrOCR**: Fine-tuned with LoRA on IAM Handwriting
- **LoRA Config**: r=16, alpha=32, 192 weights

---

### Stage 3: Action Extraction

```
┌─────────────┐     ┌─────────────────┐     ┌─────────────┐
│  OCR Text   │ --> │  Qwen2.5-7B     │ --> │   JSON      │
│             │     │  + QLoRA        │     │   Output    │
└─────────────┘     └─────────────────┘     └─────────────┘
```

- **Model**: Qwen2.5-7B-Instruct
- **Quantization**: 4-bit (NF4)
- **LoRA**: r=64, alpha=128
- **Training**: 5000 synthetic samples, 3 epochs

---

## Training Summary

| Model | Dataset | LoRA | Epochs | Key Params |
|-------|---------|------|--------|------------|
| YOLOv8 | Roboflow Whiteboard | N/A | 100 | lr=0.001 |
| TrOCR | IAM Handwriting | r=16 | 10 | warmup=500 steps |
| Qwen2.5 | Synthetic Action Items | r=64 | 3 | warmup=3%, cosine |

---

## Key Learnings

| Issue | Root Cause | Solution |
|-------|------------|----------|
| TrOCR garbage output | Expects single-line images | Use hybrid (EasyOCR detect + TrOCR read) |
| LoRA weights not loading | Missing PEFT wrapper | Apply `get_peft_model()` before loading |
| Poor OCR accuracy | Training ≠ inference data | Match training distribution to real inputs |

---

## Usage

```python
from whiteboard_ai import WhiteboardAI

# Initialize with hybrid OCR (uses trained TrOCR LoRA)
ai = WhiteboardAI(device='cuda', ocr_backend='hybrid')

# Analyze whiteboard
results = ai.analyze_whiteboard('whiteboard.jpg')

# Access results
print(results['full_text'])        # OCR text
print(results['action_items'])     # Extracted actions
```

---

## File Structure

```
whiteboard-ai/
├── whiteboard_ai.py          # Main pipeline
├── training/
│   ├── train_yolo.py         # YOLOv8 training
│   ├── train_trocr.py        # TrOCR + LoRA training
│   └── train_qwen_lora.py    # Qwen2.5 QLoRA training
└── runs/
    ├── trocr_handwriting/    # TrOCR LoRA weights
    └── qwen_action_items/    # Qwen LoRA adapter
```
