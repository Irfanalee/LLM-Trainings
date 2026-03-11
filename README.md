# LLM Trainings Repository

This repository contains projects focused on fine-tuning large language models for specialized tasks. Each folder contains implementations, datasets, and documentation for different AI training approaches.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?style=flat&logo=nvidia&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat&logo=huggingface&logoColor=black)
![Unsloth](https://img.shields.io/badge/Unsloth-QLoRA-8A2BE2?style=flat)
![Hardware](https://img.shields.io/badge/GPU-RTX_A4000_16GB-76B900?style=flat&logo=nvidia&logoColor=white)

---

## Projects

### 📁 [code-review-critic](./code-review-critic/) -- INPROGRESS

Fine-tuned Qwen2.5-Coder-7B to provide constructive code review feedback on Python code. Uses QLoRA training on RTX A4000 (16GB). Includes PR review automation script.

![Qwen](https://img.shields.io/badge/Model-Qwen2.5--Coder--7B-0066CC?style=flat)
![QLoRA](https://img.shields.io/badge/Method-QLoRA-8A2BE2?style=flat)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Unsloth](https://img.shields.io/badge/Unsloth-Fast_Training-8A2BE2?style=flat)

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

![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?style=flat&logo=opencv&logoColor=white)
![OCR](https://img.shields.io/badge/OCR-Text_Extraction-FF6F00?style=flat)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)

---

### 📁 [doc_intelligence_agent](./doc_intelligence_agent/) -- ONHOLD

Document intelligence agent for processing and understanding documents.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Agent-1C3C3C?style=flat)

---

### 📁 [JSON-Asset-Tagger](./JSON-Asset-Tagger/) -- ONHOLD

Train a specialized JSON Asset Tagger model to extract equipment details from engineering text. Includes data generation, Jupyter tutorials, and fine-tuning instructions for Qwen2.5-Coder using LoRA on Google Colab or Ollama.

![Qwen](https://img.shields.io/badge/Model-Qwen2.5--Coder-0066CC?style=flat)
![LoRA](https://img.shields.io/badge/Method-LoRA-8A2BE2?style=flat)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Local_Inference-000000?style=flat)
![Colab](https://img.shields.io/badge/Google_Colab-Training-F9AB00?style=flat&logo=googlecolab&logoColor=white)

---

### 📁 [Incidents-data-scrapper](./Incidents-data-scrapper/) -- INPROGRESS

Fine-tuned Mistral-NeMo-Minitron-8B-Instruct as a DevOps Incident Responder. Uses QLoRA (4-bit quantization + LoRA adapters) on RTX A4000 (16GB).

![Mistral](https://img.shields.io/badge/Model-Mistral--NeMo--8B-FF7000?style=flat)
![QLoRA](https://img.shields.io/badge/Method-QLoRA-8A2BE2?style=flat)
![Unsloth](https://img.shields.io/badge/Unsloth-Fast_Training-8A2BE2?style=flat)
![TRL](https://img.shields.io/badge/TRL-SFT_Trainer-FFD21E?style=flat)

| Component | Details |
|-----------|---------|
| Base Model | nvidia/Mistral-NeMo-Minitron-8B-Instruct |
| Method | QLoRA (4-bit + LoRA, rank 32) |
| Hardware | RTX A4000 16GB |

**Status:** 🔄 Training in progress

---

### 📁 [WorldGuard](./WorldGuard/) -- INPROGRESS

JEPA-inspired video world model for unsupervised CCTV anomaly detection. Trains on normal scenes only — anomalies are detected as high prediction error in latent space. No labels required.

![PyTorch](https://img.shields.io/badge/PyTorch-Raw_Training_Loop-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![JEPA](https://img.shields.io/badge/Architecture-I--JEPA-0066CC?style=flat)
![timm](https://img.shields.io/badge/timm-Vision_Encoder-5C3EE8?style=flat)
![PyAV](https://img.shields.io/badge/PyAV-Video_I%2FO-FF6F00?style=flat)
![wandb](https://img.shields.io/badge/W%26B-Experiment_Tracking-FFBE00?style=flat&logo=weightsandbiases&logoColor=black)

| Component | Details |
|-----------|---------|
| Architecture | I-JEPA (context encoder → predictor → EMA target encoder) |
| Method | Latent space prediction (no pixel reconstruction) |
| Hardware | RTX A4000 16GB |
| Framework | Raw PyTorch (no HuggingFace Trainer) |

**Status:** 🔄 Data pipeline and model architecture complete — training loop in progress

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
