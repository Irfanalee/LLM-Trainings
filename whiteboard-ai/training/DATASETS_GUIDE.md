# Whiteboard AI - Dataset Guide

## Quick Start (Easiest Path)

**Get something working TODAY:**
1. Skip YOLOv8 training - use the whole image as one region (already implemented)
2. Use pretrained TrOCR - it works well out of the box for handwriting
3. Use zero-shot Qwen2.5 with better prompting (free improvement)
4. Generate synthetic data for fine-tuning later

---

## PART 1: Region Detection Datasets (YOLOv8)

### Option A: Use Document Layout Datasets (RECOMMENDED)

These datasets have similar regions to whiteboards:

| Dataset | Size | Link | Notes |
|---------|------|------|-------|
| **DocLayNet** | 80K images | https://github.com/DS4SD/DocLayNet | IBM's document layout dataset, 11 classes |
| **PubLayNet** | 360K images | https://github.com/ibm-aur-nlp/PubLayNet | Scientific papers, 5 classes |
| **IIIT-AR-13K** | 13K images | http://cvit.iiit.ac.in/usodi/iiitar13k.php | Handwritten forms |
| **TableBank** | 417K images | https://github.com/doc-analysis/TableBank | Tables detection |

### Option B: Whiteboard-Specific (Limited)

| Dataset | Size | Link | Notes |
|---------|------|------|-------|
| **COCO Whiteboard** | ~500 images | Filter COCO for "whiteboard" | General whiteboards |
| **Lecture Videos** | Extract frames | YouTube/Coursera | Create your own |

### Option C: Custom Labeling (Best Quality)

Use **Label Studio** or **CVAT** to label your own whiteboards:
```bash
# Install Label Studio
pip install label-studio
label-studio start
```

**Recommended Classes for Whiteboards:**
- `header` - Title or heading text
- `text_block` - Main body of handwritten text
- `diagram` - Flowcharts, drawings
- `bullet_list` - Lists with bullets/numbers
- `table` - Tabular content
- `arrow` - Connecting arrows
- `sticky_note` - Post-it notes

### Download DocLayNet (Recommended)

```bash
# DocLayNet - IBM's Document Layout Dataset
git lfs install
git clone https://huggingface.co/datasets/ds4sd/DocLayNet

# Or direct download
wget https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_core.zip
unzip DocLayNet_core.zip
```

---

## PART 2: Handwriting OCR Datasets (TrOCR)

### Pretrained TrOCR is Usually Sufficient!

Microsoft's TrOCR was trained on:
- IAM Handwriting Database
- Synthetic handwriting data
- Multiple languages

**Try pretrained first before fine-tuning!**

### If Fine-Tuning Needed:

| Dataset | Size | Link | Language | Notes |
|---------|------|------|----------|-------|
| **IAM Handwriting** | 13K lines | https://fki.tic.heia-fr.ch/databases/iam-handwriting-database | English | Gold standard |
| **RIMES** | 12K pages | https://www.a2ialab.com/doku.php?id=rimes_database:start | French | Forms |
| **CVL** | 7 scripts | https://cvl.tuwien.ac.at/research/cvl-databases/an-off-line-database-for-writer-retrieval-writer-identification-and-word-spotting/ | Multi | Writers |
| **Bentham** | 11K pages | https://zenodo.org/record/44519 | English | Historical |
| **HWD+** | 20K words | https://github.com/herobd/HWD | English | Modern |

### Download IAM Dataset

```bash
# Requires registration at: https://fki.tic.heia-fr.ch/login
# After registration, download:
# - words.tgz (handwriting word images)
# - ascii/words.txt (transcriptions)

# Expected structure:
# datasets/iam/
# ├── words/
# │   └── a01/
# │       └── a01-000u/
# │           └── a01-000u-00-00.png
# └── words.txt
```

---

## PART 3: Action Item Extraction (Qwen2.5 LoRA)

### Option A: Use Synthetic Data (RECOMMENDED)

Generate meeting notes programmatically - I'll provide a script below!

### Option B: Existing Meeting Datasets

| Dataset | Size | Link | Notes |
|---------|------|------|-------|
| **AMI Meeting Corpus** | 100 hrs | https://groups.inf.ed.ac.uk/ami/corpus/ | Audio + transcripts |
| **ICSI Meeting Corpus** | 72 hrs | http://www1.icsi.berkeley.edu/Speech/mr/ | Research meetings |
| **QMSum** | 1.8K meetings | https://github.com/Yale-LILY/QMSum | Query-based summaries |
| **DialogSum** | 13K dialogues | https://github.com/cylnlp/DialogSum | Conversation summaries |

### Option C: Generate with GPT/Claude

Use an LLM to generate diverse meeting notes:

```python
# Example prompt for synthetic data
prompt = """Generate a realistic meeting note that includes:
- 3-5 discussion points
- 2-4 action items with assignees
- Deadline mentions
- Priority indicators

Make it look like handwritten whiteboard notes (informal, abbreviated).
"""
```

---

## PART 4: Quick Download Script

```bash
#!/bin/bash
# save as: download_datasets.sh

DATASET_DIR="./datasets"
mkdir -p $DATASET_DIR

# 1. DocLayNet (for region detection)
echo "Downloading DocLayNet..."
cd $DATASET_DIR
wget -c https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_core.zip
unzip -n DocLayNet_core.zip

# 2. IAM requires manual download (registration required)
echo "IAM Handwriting Database requires manual registration:"
echo "  Visit: https://fki.tic.heia-fr.ch/databases/iam-handwriting-database"

# 3. We'll generate synthetic meeting notes (see scripts/)
echo "Synthetic meeting data will be generated with the provided scripts"

echo "Done! Check $DATASET_DIR"
```

---

## Practical Recommendation

### Phase 1: Get Working Demo (Today)
1. **Skip fine-tuning** - Use your current setup
2. **Better prompting** for Qwen2.5 (see scripts/better_prompts.py)
3. **Test with sample images** - See what works

### Phase 2: Generate Synthetic Data (This Week)
1. Generate 1000+ synthetic whiteboard images
2. Generate 5000+ meeting notes with action items
3. Label a few real whiteboards manually

### Phase 3: Fine-tune Models (Next Week)
1. Train YOLOv8 on synthetic whiteboards + DocLayNet
2. Fine-tune Qwen2.5 with LoRA on action items
3. Only fine-tune TrOCR if needed

---

## Memory Requirements (RTX A4000 16GB)

| Model | Full Training | LoRA/QLoRA | Inference |
|-------|--------------|------------|-----------|
| YOLOv8n | 2-4 GB | N/A | 1 GB |
| TrOCR-large | 8-10 GB | 4-6 GB | 3 GB |
| Qwen2.5-7B | 28+ GB | 8-12 GB | 4-6 GB (4-bit) |

**Your A4000 can handle all of these with QLoRA!**
