# 📊 Complete Dataset Guide for Whiteboard Meeting Notes AI

## 🎯 Overview

This guide provides **direct links and instructions** to download datasets for all three components of your whiteboard AI system.

---

## 1️⃣ WHITEBOARD REGION DETECTION (YOLOv8)

### **Option A: Roboflow Whiteboard Datasets** ⭐ RECOMMENDED

**Best Option - 1,124 images:**
- **Dataset**: Whiteboard Detection
- **Link**: https://universe.roboflow.com/whiteboard-kw2vt/whiteboard-detect
- **Size**: 1,124 labeled images
- **Format**: YOLO, COCO JSON, Pascal VOC
- **License**: Public Domain
- **Classes**: whiteboard, board, people

**Alternative - 1,012 images:**
- **Dataset**: WhiteBoard Detection by BIIT
- **Link**: https://universe.roboflow.com/biit/whiteboard-detection
- **Size**: 1,012 images
- **Classes**: WhiteBoard, board, people
- **License**: CC BY 4.0

**How to Download:**
```bash
# Install roboflow
pip install roboflow

# Download dataset (Python)
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")  # Get free API key from roboflow.com
project = rf.workspace("whiteboard-kw2vt").project("whiteboard-detect")
dataset = project.version(1).download("yolov8")
```

### **Option B: COCO Dataset (General Objects)**

If you want to train on general object detection including whiteboards:
- **Link**: https://cocodataset.org/#download
- **Classes**: 80 objects (includes some office equipment)
- **Size**: 118K training images
- **Note**: Not whiteboard-specific, but good for general detection

---

## 2️⃣ HANDWRITING OCR (TrOCR Fine-tuning)

### **Option A: IAM Handwriting Database** ⭐ RECOMMENDED

**Official Source:**
- **Link**: https://fki.tic.heia-fr.ch/databases/download-the-iam-handwriting-database
- **Registration Required**: Yes (free, academic use)
- **Size**: 115,320 word images from 657 writers
- **Format**: Images + text transcriptions
- **License**: Free for non-commercial research

**What to Download:**
1. `words.tgz` - Individual word images (13 GB)
2. `words.txt` - Transcriptions
3. Or `lines.tgz` - Line-level images (9 GB)

**Hugging Face Mirror (Easier):**
```python
from datasets import load_dataset

# Load IAM dataset directly
dataset = load_dataset("Teklia/IAM-line")

# Access
train_data = dataset['train']
for item in train_data:
    image = item['image']
    text = item['text']
```

### **Option B: RIMES Dataset (French Handwriting)**

- **Link**: http://www.a2ialab.com/doku.php?id=rimes_database:data:icdar2011:line:start
- **Size**: 12,723 line images
- **Language**: French
- **Use Case**: Multi-language support

### **Option C: Kaggle IAM Dataset**

- **Link**: https://www.kaggle.com/datasets/nibinv23/iam-handwriting-word-database
- **Size**: 115K word images
- **Format**: Pre-processed and ready to use
- **No registration needed** (just Kaggle account)

---

## 3️⃣ ACTION ITEM EXTRACTION (LLM Fine-tuning)

### **Option A: AMI Meeting Corpus** ⭐ RECOMMENDED

**For Meeting Transcripts:**
- **Link**: https://groups.inf.ed.ac.uk/ami/corpus/
- **Size**: 171 meeting transcripts
- **Annotations**: Summaries, action items (indirect)
- **Format**: XML transcripts
- **License**: Free for research

**How to Use:**
- Action items are linked to abstractive summaries
- 101 meetings with 381 action items
- Download: Meeting transcripts + annotations

### **Option B: MeetingBank Dataset** ⭐ NEW & LARGE

**Best Modern Dataset:**
```python
from datasets import load_dataset

# Load MeetingBank
meetingbank = load_dataset("huuuyeah/meetingbank")

train_data = meetingbank['train']
test_data = meetingbank['test']

# Each meeting has:
# - transcript (full text)
# - summary (with action items)
# - metadata (agenda, etc.)
```

**Details:**
- **Size**: 1,366 meetings, 3,579 hours of video
- **Source**: City council meetings (public data)
- **Format**: Hugging Face dataset
- **Includes**: Transcripts, summaries, agendas

### **Option C: ICSI Meeting Corpus**

- **Link**: https://groups.inf.ed.ac.uk/ami/icsi/
- **Size**: 75 meetings
- **Note**: Smaller but high quality
- **Annotations**: Direct action item labels (18 meetings)

### **Option D: Synthetic Data Generation** ⭐ EASIEST START

**Generate your own dataset using GPT/Claude:**

```python
# Example: Generate synthetic meeting notes
import anthropic

client = anthropic.Anthropic(api_key="your-key")

prompt = """Generate a realistic meeting transcript with action items.
Include:
- TODO items with assignees
- Deadlines
- Clear task descriptions

Format as JSON:
{
  "transcript": "full meeting text",
  "action_items": [
    {"task": "...", "assignee": "...", "deadline": "..."}
  ]
}
"""

# Generate 100-1000 examples
for i in range(100):
    response = client.messages.create(...)
    # Save to dataset
```

---

## 📥 QUICK START DOWNLOAD SCRIPT

Save this as `download_datasets.py`:

```python
"""
Download all datasets for Whiteboard AI
Run: python download_datasets.py
"""

from datasets import load_dataset
from roboflow import Roboflow
import os

# Create data directory
os.makedirs("data", exist_ok=True)

print("="*60)
print("DOWNLOADING DATASETS FOR WHITEBOARD AI")
print("="*60)

# 1. WHITEBOARD DETECTION
print("\n1. Downloading Whiteboard Detection Dataset...")
print("Please sign up at roboflow.com and get your API key")
print("Then uncomment and run the roboflow code")

# rf = Roboflow(api_key="YOUR_KEY_HERE")
# project = rf.workspace("whiteboard-kw2vt").project("whiteboard-detect")
# dataset = project.version(1).download("yolov8", location="data/whiteboards")

# 2. IAM HANDWRITING
print("\n2. Downloading IAM Handwriting Dataset...")
try:
    iam_dataset = load_dataset("Teklia/IAM-line", cache_dir="data/iam")
    print(f"✅ Downloaded {len(iam_dataset['train'])} training samples")
except Exception as e:
    print(f"❌ Error: {e}")
    print("Try: pip install datasets")

# 3. MEETINGBANK
print("\n3. Downloading MeetingBank Dataset...")
try:
    meeting_dataset = load_dataset("huuuyeah/meetingbank", cache_dir="data/meetingbank")
    print(f"✅ Downloaded {len(meeting_dataset['train'])} meetings")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n"+"="*60)
print("DOWNLOAD COMPLETE!")
print("="*60)
print("\nDatasets saved to ./data/")
print("\nNext steps:")
print("1. Prepare data for training")
print("2. Fine-tune models")
print("3. Evaluate performance")
```

---

## 🔧 DATA PREPARATION PIPELINE

### **For YOLOv8 (Whiteboard Detection)**

```python
# Data is already in YOLO format from Roboflow
# Structure:
# data/whiteboards/
#   ├── train/
#   │   ├── images/
#   │   └── labels/
#   ├── valid/
#   └── test/

# Create data.yaml
with open("data/whiteboards/data.yaml", "w") as f:
    f.write("""
train: train/images
val: valid/images
test: test/images

nc: 3
names: ['whiteboard', 'board', 'people']
""")

# Train YOLOv8
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.train(
    data='data/whiteboards/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)
```

### **For TrOCR (Handwriting OCR)**

```python
from datasets import load_dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Load IAM dataset
dataset = load_dataset("Teklia/IAM-line")

# Prepare for training
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

def preprocess(examples):
    images = [img.convert("RGB") for img in examples['image']]
    pixel_values = processor(images, return_tensors="pt").pixel_values
    labels = processor.tokenizer(
        examples['text'],
        padding="max_length",
        max_length=128
    ).input_ids
    return {"pixel_values": pixel_values, "labels": labels}

train_dataset = dataset['train'].map(preprocess, batched=True)
```

### **For Qwen2.5 (Action Items)**

```python
from datasets import load_dataset
import json

# Load MeetingBank
dataset = load_dataset("huuuyeah/meetingbank")

# Convert to instruction format for fine-tuning
def create_training_example(meeting):
    return {
        "messages": [
            {
                "role": "system",
                "content": "Extract action items from meeting transcripts."
            },
            {
                "role": "user",
                "content": f"Extract action items from this meeting:\n\n{meeting['transcript']}"
            },
            {
                "role": "assistant",
                "content": meeting['summary']  # Contains action items
            }
        ]
    }

# Save as JSONL for training
with open("data/action_items_train.jsonl", "w") as f:
    for item in dataset['train']:
        f.write(json.dumps(create_training_example(item)) + "\n")
```

---

## 📊 DATASET STATISTICS

| Dataset | Size | Task | Download Time | Disk Space |
|---------|------|------|---------------|------------|
| Whiteboard Detection | 1,124 images | Region Detection | 5 min | ~500 MB |
| IAM Handwriting | 115K words | OCR | 10 min | ~2 GB |
| MeetingBank | 1,366 meetings | Action Extraction | 30 min | ~5 GB |
| **TOTAL** | - | - | **~45 min** | **~7.5 GB** |

---

## 🚀 ALTERNATIVE: START WITHOUT TRAINING

You can use the **pretrained models** without any training:

1. **YOLOv8**: Use `yolov8n.pt` (already trained on COCO)
2. **TrOCR**: Use `microsoft/trocr-large-handwritten` (already trained on IAM)
3. **Qwen2.5**: Use prompt engineering (zero-shot) first

**This is what your current code does!** Training is optional for improvements.

---

## 💡 RECOMMENDED APPROACH

### **Phase 1: Test with Pretrained Models** (Day 1)
- ✅ Use existing models
- ✅ Test on real whiteboard photos
- ✅ Evaluate performance

### **Phase 2: Collect Real Data** (Week 1)
- 📸 Take 50-100 whiteboard photos yourself
- ✍️ Label regions manually (Roboflow is free)
- 📝 Create meeting notes with action items

### **Phase 3: Fine-tune (Optional)** (Week 2+)
- 🎯 YOLOv8 on your whiteboard photos
- ✍️ TrOCR if handwriting is very different
- 🤖 Qwen2.5 with LoRA on your meeting format

---

## 📚 USEFUL LINKS

- **Roboflow Universe**: https://universe.roboflow.com/
- **Hugging Face Datasets**: https://huggingface.co/datasets
- **COCO Dataset**: https://cocodataset.org/
- **IAM Database**: https://fki.tic.heia-fr.ch/databases/iam-handwriting-database
- **MeetingBank**: https://meetingbank.github.io/

---

## ❓ NEED HELP?

Use this prompt in VS Code Claude:
> "I've downloaded [dataset name]. How do I prepare it for training [model name]?"

Happy training! 🚀
