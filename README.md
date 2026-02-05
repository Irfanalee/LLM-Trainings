# LLM Trainings Repository

This repository contains projects focused on fine-tuning large language models for specialized tasks. Each folder contains implementations, datasets, and documentation for different AI training approaches.

---

## Projects

### 📁 [code-review-critic](./code-review-critic/) -- INPROGRESS


Fine-tuned Qwen2.5-Coder-7B to provide constructive code review feedback on Python code. Uses QLoRA training on RTX A4000 (16GB). Includes PR review automation script.

| Component | Details |
|-----------|---------|
| Base Model | Qwen2.5-Coder-7B-Instruct |
| Method | QLoRA (4-bit + LoRA) |
| Training Data | 8,275 examples (GitHub PR comments + synthetic) |
| Hardware | RTX A4000 16GB |

**Status:** ✅ Training complete, model working

---

### 📁 [whiteboard-ai](./whiteboard-ai/) -- ONHOLD

Computer vision pipeline for whiteboard content extraction and understanding. Combines object detection with OCR and LLM processing.

---

### 📁 [doc_intelligence_agent](./doc_intelligence_agent/) -- ONHOLD

Document intelligence agent for processing and understanding documents.

---

### 📁 [JSON-Asset-Tagger](./JSON-Asset-Tagger/) --ONHOLD

Train a specialized JSON Asset Tagger model to extract equipment details from engineering text. Includes data generation, Jupyter tutorials, and fine-tuning instructions for Qwen2.5-Coder using LoRA on Google Colab or Ollama.

---

## Getting Started

1. Navigate to the specific folder of interest
2. Review the corresponding README for detailed instructions
3. Install dependencies from `requirements.txt`
4. Follow the tutorial notebooks or scripts

---

## Environment

- GPU: NVIDIA RTX A4000 16GB
- OS: Ubuntu Linux
- Python: 3.12
- Key packages: unsloth, transformers, trl, peft, bitsandbytes

---

## License

[Add license information here]

## Notes

This is an ongoing project. More folders and modules will be added as the repository grows.
