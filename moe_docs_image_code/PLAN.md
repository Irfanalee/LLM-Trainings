# Document Intelligence MoE - Weekly Plan

## Project Overview

**Goal:** Fine-tune Qwen3-30B-A3B (real MoE architecture) for document intelligence — extracting structured data from invoices, contracts, and general documents.

**Model:** Qwen3-30B-A3B (30B total params, 3B active per token)
**Training:** QLoRA with Unsloth's optimized MoE Triton kernels
**Hardware:** NVIDIA RTX A4000 16GB

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Document Text / Image                       │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Qwen3-30B-A3B (MoE Architecture)                │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │         Learned Router (selects top-k experts)         │ │
│  └────────────────────────────────────────────────────────┘ │
│                              │                               │
│       ┌──────────┬──────────┼──────────┬──────────┐        │
│       ▼          ▼          ▼          ▼          ▼        │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐   │
│  │Expert 1│ │Expert 2│ │Expert 3│ │  ...   │ │Expert N│   │
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘   │
│                                                              │
│            (Only 3B parameters active per token)            │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
                ┌─────────────────────────────┐
                │  Structured JSON + Summary   │
                └─────────────────────────────┘
```

### Why Real MoE vs Prompt Routing?

| Aspect | Prompt Routing | Real MoE (Our Approach) |
|--------|---------------|------------------------|
| Experts | Different system prompts | Neural network experts |
| Routing | Manual classifier | Learned by model |
| Training | No fine-tuning | Fine-tune with Unsloth |
| Quality | Limited by prompts | Learns from data |
| Efficiency | Full model active | Only 3B/30B active |
| LinkedIn Value | "I built a router" | "I trained an MoE" |

---

## Datasets

| Dataset | Type | Size | Use For |
|---------|------|------|---------|
| **CORD** | Receipts/Invoices | 11K+ | Invoice extraction |
| **CUAD** | Legal Contracts | 510 contracts, 13K labels | Contract analysis |
| **DocVQA** | General Documents | 50K Q&A | Document Q&A |

**Location:** `/repos/LLM-Trainings/moe_docs_image_code/data/datasets/`

---

## Week Schedule

### Day 1 (Monday): Setup & Data Preparation

**Morning:**
- [x] Download datasets (CORD, CUAD, DocVQA)
- [ ] Verify dataset structure and contents
- [ ] Update paths in `config.py`

**Afternoon:**
- [ ] Run `prepare_training_data.py`
- [ ] Verify training data format
- [ ] Check data distribution across document types

**Deliverable:** Training data ready in `data/training/`

**Commands:**
```bash
cd /repos/LLM-Trainings/moe_docs_image_code

# Prepare training data
python prepare_training_data.py

# Check output
ls -la data/training/
cat data/training/stats.json
```

---

### Day 2 (Tuesday): Model Setup & First Training Run

**Morning:**
- [ ] Install/update Unsloth with MoE support
- [ ] Test model loading (Qwen3-30B-A3B in 4-bit)
- [ ] Verify VRAM usage (~14GB expected)

**Afternoon:**
- [ ] Start training run
- [ ] Monitor loss curves
- [ ] Check for OOM or other issues

**Deliverable:** Training running, first checkpoint saved

**Commands:**
```bash
# Update Unsloth
pip install --upgrade unsloth

# Start training
python train_moe.py

# Monitor GPU
watch -n 1 nvidia-smi
```

---

### Day 3 (Wednesday): Training Completion & Evaluation

**Morning:**
- [ ] Training should complete (~2-4 hours for 2 epochs)
- [ ] Check final loss values
- [ ] Save best checkpoint

**Afternoon:**
- [ ] Run `test_moe.py` with test cases
- [ ] Evaluate invoice extraction quality
- [ ] Evaluate contract analysis quality
- [ ] Evaluate general Q&A quality

**Deliverable:** Trained model, initial quality assessment

**Commands:**
```bash
# Test the model
python test_moe.py
```

---

### Day 4 (Thursday): Iteration & Improvement

**Morning:**
- [ ] Analyze failure cases
- [ ] Identify data quality issues
- [ ] Adjust prompts if needed

**Afternoon:**
- [ ] Optional: Generate synthetic data for weak areas
- [ ] Optional: Re-train with improved data
- [ ] Test again

**Deliverable:** Improved model or clear understanding of limitations

---

### Day 5 (Friday): Inference Pipeline & Demo

**Morning:**
- [ ] Create inference script (`inference.py`)
- [ ] Add support for different document types
- [ ] Optimize inference speed

**Afternoon:**
- [ ] Build Gradio demo (`app.py`)
- [ ] Test with real documents
- [ ] Fix UI/UX issues

**Deliverable:** Working demo app

---

### Day 6 (Saturday): Export & Deployment

**Morning:**
- [ ] Export to GGUF format for llama.cpp/Ollama
- [ ] Test quantized model quality
- [ ] Benchmark inference speed

**Afternoon:**
- [ ] Create Ollama Modelfile
- [ ] Test local deployment
- [ ] Document deployment steps

**Deliverable:** Deployable model in GGUF format

**Commands:**
```bash
# Export to GGUF
python export_gguf.py

# Test with Ollama
ollama create doc-intel -f Modelfile
ollama run doc-intel
```

---

### Day 7 (Sunday): Documentation & Showcase

**Morning:**
- [ ] Write comprehensive README
- [ ] Create architecture diagrams
- [ ] Document training process and results
- [ ] Prepare sample outputs

**Afternoon:**
- [ ] Create GitHub repository
- [ ] Record demo video/GIF
- [ ] Write LinkedIn post
- [ ] Optional: Deploy to Hugging Face Spaces

**Deliverable:** Published project ready for showcase

---

## Project Files

### Core Files (Ready ✅)
| File | Purpose |
|------|---------|
| `config.py` | Configuration, paths, system prompts |
| `prepare_training_data.py` | Convert datasets → training format |
| `train_moe.py` | Unsloth QLoRA training |
| `test_moe.py` | Test fine-tuned model |
| `download_datasets.py` | Dataset downloader |

### To Create This Week
| File | Day | Purpose |
|------|-----|---------|
| `inference.py` | Day 5 | Production inference |
| `app.py` | Day 5 | Gradio demo |
| `export_gguf.py` | Day 6 | GGUF export |
| `Modelfile` | Day 6 | Ollama config |
| `README.md` | Day 7 | Documentation |

### Directory Structure
```
/repos/LLM-Trainings/moe_docs_image_code/
├── config.py
├── prepare_training_data.py
├── train_moe.py
├── test_moe.py
├── download_datasets.py
├── PLAN.md
├── CLAUDE_CODE_PROMPT.md
├── data/
│   ├── datasets/           # Downloaded datasets
│   │   ├── cord/
│   │   ├── cuad/
│   │   └── docvqa/
│   └── training/           # Processed training data
│       ├── train.jsonl
│       ├── eval.jsonl
│       └── stats.json
└── output/                 # Training outputs
    └── lora_adapters/      # Saved LoRA weights
```

---

## Technical Specifications

### Model
| Spec | Value |
|------|-------|
| Base Model | Qwen3-30B-A3B |
| Architecture | Mixture of Experts |
| Total Params | 30B |
| Active Params | 3B per token |
| Experts | 128 (top-k routing) |

### Training
| Spec | Value |
|------|-------|
| Method | QLoRA (4-bit + LoRA) |
| LoRA Rank | 32 |
| LoRA Alpha | 64 |
| Batch Size | 1 |
| Gradient Accumulation | 16 |
| Effective Batch | 16 |
| Epochs | 2 |
| Learning Rate | 2e-4 |

### VRAM Usage
| Component | VRAM |
|-----------|------|
| Model (4-bit) | ~12GB |
| LoRA adapters | ~1GB |
| Optimizer | ~2GB |
| **Total** | **~14-15GB** ✅ |

---

## Success Metrics

| Task | Metric | Target |
|------|--------|--------|
| Invoice extraction | Field accuracy | >85% |
| Contract analysis | Clause ID | >80% |
| General Q&A | Answer accuracy | >80% |
| Training time | Hours | <4 |
| Inference | Seconds/doc | <5 |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| OOM during training | Reduce batch to 1, increase grad accumulation |
| Slow training | Enable packing=True, update Unsloth |
| Poor quality | Check data format, try 3 epochs |
| Model won't load | Update Unsloth: `pip install --upgrade unsloth` |

---

## LinkedIn Post Draft

```
🚀 Fine-tuned a Mixture of Experts model for Document Intelligence!

What it does:
📋 Invoices → Extracts vendor, items, amounts, dates
📝 Contracts → Identifies parties, terms, obligations  
📄 Documents → Answers questions, provides summaries

The cool part:
• Qwen3-30B-A3B: 30B params, only 3B active per token
• Unsloth's MoE Triton kernels: 12x faster training
• Runs on a single RTX A4000 (16GB)

Training took just 3 hours. The MoE architecture naturally specializes different experts for different document types.

Datasets: CORD (invoices), CUAD (contracts), DocVQA (general)

#AI #MixtureOfExperts #DocumentAI #LLM #Unsloth
```

---

## End of Week Checklist

- [ ] Training data prepared
- [ ] Model trained (2 epochs)
- [ ] Quality tested on all document types
- [ ] Demo app working
- [ ] GGUF export done
- [ ] GitHub repo published
- [ ] LinkedIn post published

---

**Let's build! 🚀**
