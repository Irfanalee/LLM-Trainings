# Code Review Critic

Fine-tuned LLM that provides constructive, actionable code review feedback on Python code. Identifies bugs, potential issues, and improvements — not just style nits.

## Features

- Reviews Python code for bugs, issues, and improvements
- Trained on 8,275 real GitHub PR review comments
- Runs on GPU (training) or CPU (inference with GGUF)
- Includes script to review real GitHub PRs

## Quick Start

```bash
# Test with sample code
python test_model.py

# Review a real GitHub PR (GPU)
python review_pr.py https://github.com/owner/repo/pull/123

# Review a real GitHub PR (CPU via Ollama)
python review_pr_ollama.py https://github.com/owner/repo/pull/123
```

## Example Output

**Input:**
```python
def get_user(user_id):
    user = db.query(User).filter(id=user_id).first()
    return user.name
```

**Review:**
```
return user.name if user else None
```

## Architecture

| Component | Choice |
|-----------|--------|
| Base Model | Qwen2.5-Coder-7B-Instruct |
| Training Method | QLoRA (4-bit + LoRA adapters) |
| Training Hardware | RTX A4000 16GB |
| Training Data | 8,275 examples |
| Export Format | Merged 16-bit → GGUF Q4 |

## Project Structure

```
code-review-critic/
├── data/processed/          # Training data (JSONL)
├── output/
│   ├── checkpoint-1036/     # Training checkpoint
│   ├── merged_model_v2/     # Working model (15GB)
│   └── gguf/                # Quantized GGUF model
├── train.py                 # QLoRA fine-tuning
├── test_model.py            # Test with samples
├── review_pr.py             # Review GitHub PRs (GPU)
├── review_pr_ollama.py      # Review GitHub PRs (CPU/Ollama)
├── quantize_cpu.py          # CPU-based GGUF quantization
├── quantize_gpu.py          # GPU-based GGUF quantization
├── Modelfile                # Ollama model definition
├── IMPROVEMENTS.md          # Future improvements
└── CLAUDE_CODE_CONTEXT.md   # Full project context
```

## Training

```bash
# Install dependencies
pip install -r requirements-train.txt

# Train (~2 hours on RTX A4000)
python train.py
```

### Training Config

```python
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
MAX_SEQ_LENGTH = 1536
LORA_R = 64
LORA_ALPHA = 64
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 16
LEARNING_RATE = 2e-4
NUM_EPOCHS = 2
```

### Training Metrics

| Metric | Value |
|--------|-------|
| Train examples | 8,275 |
| Eval examples | 780 |
| Final eval loss | 0.845 |
| Runtime | ~2 hours |

## Usage

### Test with Sample Code

```bash
python test_model.py
```

### Review a GitHub PR (GPU)

```bash
python review_pr.py https://github.com/HKUDS/nanobot/pull/109
```

### Review a GitHub PR (CPU/Ollama)

No GPU required. Uses the Ollama model via HTTP API.

```bash
# Ensure Ollama is running and model is loaded
ollama list

# Review PR
python review_pr_ollama.py https://github.com/HKUDS/nanobot/pull/109
```

### Re-merge Model (if needed)

```bash
python -c "
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained('./output/checkpoint-1036', max_seq_length=1536, load_in_4bit=True)
model.save_pretrained_merged('./output/merged_model_v2', tokenizer, save_method='merged_16bit')
"
```

## Quantization (for CPU deployment)

Convert to GGUF format for running on CPU with Ollama:

### Option 1: CPU-based (Recommended for 16GB GPU)
```bash
# Install prerequisites
sudo apt update && sudo apt install -y cmake build-essential

# Quantize (~20-30 min, uses CPU only)
python quantize_cpu.py
```

### Option 2: GPU-based (Requires 24GB+ VRAM)
```bash
# Faster but needs more VRAM
python quantize_gpu.py
```

This creates `./output/gguf/code-review-critic-q4_k_m.gguf` (~4-5GB).

### Ollama Setup

```bash
# Create Modelfile
cat > Modelfile << 'EOF'
FROM ./output/gguf/code-review-critic-q4_k_m.gguf

TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|}
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""

SYSTEM "You are an expert code reviewer. Analyze the provided Python code and give constructive, specific feedback."

PARAMETER stop "<|im_end|>"
EOF

# Create model
ollama create code-review-critic -f Modelfile

# Test
ollama run code-review-critic "Review: def get_user(id): return db.query(User).first().name"
```

## Next Steps

- [ ] Create Docker image with Ollama
- [ ] Build GitHub Action for automated PR reviews
- [ ] Improve model quality (see IMPROVEMENTS.md)

## PR Review Scripts

| Script | Hardware | Dependencies | Speed |
|--------|----------|--------------|-------|
| `review_pr.py` | GPU (16GB VRAM) | unsloth, torch | Fast |
| `review_pr_ollama.py` | CPU only | requests | Slower but portable |

## Requirements

- GPU: NVIDIA with 16GB+ VRAM (for training)
- Python: 3.12
- Key packages: unsloth, transformers, trl, peft, bitsandbytes

## License

[Add license]
