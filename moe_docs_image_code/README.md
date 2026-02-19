# Document Intelligence — Fine-tuned gpt-oss-20b

Fine-tuning **OpenAI's gpt-oss-20b** (open-sourced GPT model) for structured document extraction using **Unsloth QLoRA** on a single NVIDIA RTX A4000 (16GB).

The model extracts structured JSON data from three document types: invoices/receipts, legal contracts, and general document Q&A.

---

## What It Does

| Document Type | Input | Output |
|---|---|---|
| **Invoice / Receipt** | Receipt or invoice text | Vendor, date, line items, subtotal, tax, total |
| **Legal Contract** | Contract excerpt | Parties, effective date, key terms, obligations, termination |
| **General Document** | Document + question | Direct answer with supporting context |

---

## Model Details

| Property | Value |
|---|---|
| Base model | `unsloth/gpt-oss-20b` (OpenAI open-source release) |
| Fine-tuning method | QLoRA (4-bit quantization + LoRA adapters) |
| Training framework | Unsloth + TRL SFT |
| Hardware | NVIDIA RTX A4000 16GB |
| Training date | February 18, 2026 |

---

## Training Configuration

| Parameter | Value |
|---|---|
| LoRA rank (r) | 32 |
| LoRA alpha | 64 |
| LoRA dropout | 0.05 |
| LoRA target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Epochs | 2 |
| Batch size | 1 |
| Gradient accumulation | 16 (effective batch = 16) |
| Learning rate | 2e-4 |
| LR scheduler | Cosine |
| Warmup ratio | 0.05 |
| Optimizer | AdamW 8-bit |
| Max sequence length | 1536 |
| Precision | bfloat16 |

---

## Training Data

| Dataset | Source | Examples | Document Type |
|---|---|---|---|
| [CORD v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) | Naver Clova | 1,000 | Receipts / Invoices |
| [CUAD](https://huggingface.co/datasets/theatticusproject/cuad-qa) | Atticus Project | 5,000 | Legal Contracts |
| [DocVQA](https://huggingface.co/datasets/HuggingFaceM4/DocumentVQA) | HuggingFace M4 | 5,000 | General Document Q&A |
| **Total** | | **11,000** | |

Train/eval split: 90/10 → **9,900 train / 1,100 eval**

All examples are formatted as ChatML conversations with task-specific system prompts. Images are not included (text-only training).

---

## Framework Versions

| Library | Version |
|---|---|
| Unsloth | 2026.2.1 |
| TRL | 0.24.0 |
| Transformers | 4.57.6 |
| PyTorch | 2.10.0+cu128 |
| CUDA | 12.8 |

---

## Output Files

```
output/
├── lora_adapters/          # LoRA adapter weights (88MB) — use with base model
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer files
├── checkpoint-1100/        # Training checkpoint
├── checkpoint-1200/        # Training checkpoint
├── checkpoint-1238/        # Final checkpoint
└── gguf/
    └── doc-intel_gguf/
        ├── gpt-oss-20b.MXFP4.gguf   # Quantized model (13GB, MXFP4)
        └── Modelfile                  # Ollama deployment config
```

---

## Project Scripts

| Script | Purpose |
|---|---|
| `prepare_training_data.py` | Converts CORD, CUAD, DocVQA datasets to JSONL training format |
| `train_moe.py` | QLoRA fine-tuning with Unsloth |
| `test_moe.py` | Runs test cases + interactive mode against the trained model |
| `export_gguf.py` | Exports trained model to GGUF format for Ollama / llama.cpp |
| `add_sroie.py` | Appends additional CORD (val/test) examples to training data |

---

## Quickstart

### Run with Python (LoRA adapters)

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="output/lora_adapters",
    max_seq_length=1536,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
```

### Run with Ollama (GGUF)

```bash
ollama create doc-intel -f output/gguf/doc-intel_gguf/Modelfile
ollama run doc-intel "Extract all information from this receipt: ..."
```

### Run test suite

```bash
python3 test_moe.py
```

---

## Example Output

**Input (Invoice):**
```
COFFEE HOUSE — Date: 2024-02-15
Cappuccino $4.50 | Croissant $3.25 | Latte $5.00
Subtotal: $12.75 | Tax: $1.02 | Total: $13.77
```

**Output:**
```json
{
  "vendor": "Coffee House",
  "date": "2024-02-15",
  "items": [
    {"name": "Cappuccino", "count": 1, "price": "4.50"},
    {"name": "Croissant",  "count": 1, "price": "3.25"},
    {"name": "Latte",      "count": 1, "price": "5.00"}
  ],
  "subtotal": "12.75",
  "tax": "1.02",
  "total": "13.77"
}
```
