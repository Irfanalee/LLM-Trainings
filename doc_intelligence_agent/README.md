# 🔍 Document Intelligence Agent

**A local AI system that understands documents and answers questions about them.**

Built with Florence-2 (vision) + Qwen2.5 (reasoning), running entirely on your GPU.

![Demo](assets/demo.gif) <!-- Add your demo gif here -->

---

## ✨ Features

- **📝 Document OCR** - Extract text from any document, scan, or image
- **🤔 Visual Q&A** - Ask natural language questions about documents
- **📊 Structured Extraction** - Pull specific fields into JSON format
- **📋 Smart Summarization** - Generate concise or detailed summaries
- **🎯 Fine-tunable** - Train on your own documents for domain expertise

## 🖥️ Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 12GB VRAM | 16GB VRAM |
| RAM | 16GB | 32GB |
| Storage | 30GB free | 50GB free |

**Tested on:** NVIDIA RTX A4000 (16GB), RTX 3090, RTX 4080

---

## 🚀 Quick Start

### 1. Clone and Setup Environment

```bash
cd doc_intelligence_agent
chmod +x setup_environment.sh
./setup_environment.sh
```

### 2. Activate Environment

```bash
source venv/bin/activate
```

### 3. Verify GPU Setup

```bash
python src/test_gpu.py
```

### 4. Download Models

```bash
python src/download_models.py
```

### 5. Run Demo

```bash
python demos/gradio_app.py
```

Then open http://localhost:7860 in your browser.

---

## 📖 Usage Examples

### Python API

```python
from src.document_agent import DocumentIntelligenceAgent
from PIL import Image

# Initialize agent
agent = DocumentIntelligenceAgent()

# Load document
image = Image.open("invoice.pdf")

# Ask questions
answer = agent.ask_question(image, "What is the total amount due?")
print(answer)

# Extract structured data
schema = {
    "invoice_number": "The invoice ID",
    "total": "Total amount",
    "due_date": "Payment due date"
}
data = agent.extract_structured_data(image, schema)
print(data)

# Generate summary
summary = agent.summarize(image, style="executive")
print(summary)
```

### Command Line

```bash
# Quick test
python src/document_agent.py

# Launch web UI
python demos/gradio_app.py

# Fine-tune on your data
python src/fine_tune.py --data-dir my_documents/ --epochs 3
```

---

## 🎯 Fine-Tuning on Your Documents

### 1. Prepare Your Data

Create `data/raw/annotations.json`:

```json
[
    {
        "document": "my_invoice.png",
        "conversations": [
            {"role": "user", "content": "What is the PO number?"},
            {"role": "assistant", "content": "The PO number is PO-2024-00123."}
        ]
    }
]
```

### 2. Run Fine-Tuning

```bash
python src/fine_tune.py --data-dir data/raw --epochs 3
```

### 3. Use Your Fine-Tuned Model

```python
agent = DocumentIntelligenceAgent(
    llm_model="models/final/document_agent_TIMESTAMP"
)
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Document Intelligence Agent               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────────┐    ┌──────────────┐    ┌─────────────┐  │
│  │   Document    │───▶│  Florence-2  │───▶│   Qwen2.5   │  │
│  │  (PDF/Image)  │    │   (Vision)   │    │ (Reasoning) │  │
│  └───────────────┘    └──────────────┘    └─────────────┘  │
│                              │                    │         │
│                              ▼                    ▼         │
│                     ┌──────────────┐    ┌─────────────┐    │
│                     │  OCR Text    │    │  Answers    │    │
│                     │  Captions    │    │  JSON Data  │    │
│                     │  Detections  │    │  Summaries  │    │
│                     └──────────────┘    └─────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Models Used:**
- **Florence-2-large** (770M params) - Microsoft's vision-language model
- **Qwen2.5-7B-Instruct** (7B params, 4-bit quantized) - Alibaba's instruction-tuned LLM

---

## 📁 Project Structure

```
doc_intelligence_agent/
├── setup_environment.sh     # One-click setup
├── src/
│   ├── test_gpu.py         # GPU verification
│   ├── download_models.py  # Model downloader
│   ├── document_agent.py   # Core agent class
│   └── fine_tune.py        # Fine-tuning script
├── demos/
│   └── gradio_app.py       # Web UI demo
├── data/
│   ├── raw/                # Your documents
│   ├── processed/          # Preprocessed data
│   └── train/              # Training data
├── models/
│   ├── checkpoints/        # Training checkpoints
│   └── final/              # Fine-tuned models
└── notebooks/              # Jupyter experiments
```

---

## 🎥 Creating Your LinkedIn Demo

### Recording Tips

1. **Show the problem first** - Upload a complex document
2. **Demonstrate the magic** - Ask a natural question, get instant answer
3. **Highlight the value** - Show structured data extraction
4. **Mention it's local** - "Running entirely on my GPU, no cloud APIs"

### Suggested Demo Flow

1. Upload a multi-page invoice or contract
2. Ask: "What is the total amount and when is it due?"
3. Extract structured data with custom fields
4. Generate an executive summary
5. Show the Gradio UI in action

### LinkedIn Post Template

```
🚀 Built a Document Intelligence Agent that runs entirely locally!

No cloud APIs. No data leaving my machine. Just my GPU doing the work.

What it does:
📝 Extracts text from any document (OCR)
🤔 Answers questions about documents naturally  
📊 Pulls structured data into JSON
📋 Generates summaries on demand

Tech stack:
• Florence-2 for vision understanding
• Qwen2.5 for reasoning (4-bit quantized)
• Runs on a single RTX A4000 (16GB)
• Fine-tunable on custom documents

The best part? I can fine-tune it on my company's specific document formats.

[Link to demo video]

#AI #MachineLearning #DocumentProcessing #LocalAI #DeepLearning
```

---

## 🐛 Troubleshooting

### "CUDA out of memory"

- Reduce batch size in fine-tuning
- Ensure 4-bit quantization is enabled
- Close other GPU applications

### "Model download fails"

- Check internet connection
- Verify disk space (need ~30GB)
- Try downloading models separately

### "Gradio not launching"

- Check if port 7860 is available
- Try: `python demos/gradio_app.py --server-port 8080`

---

## 📄 License

MIT License - feel free to use for personal and commercial projects.

---

## 🙏 Acknowledgments

- [Microsoft Florence-2](https://huggingface.co/microsoft/Florence-2-large)
- [Alibaba Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)

---

**Built with ❤️ for the AI community**
