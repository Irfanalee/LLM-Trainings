"""
Prepare Training Data for Document Intelligence MoE
Converts CORD, CUAD, and DocVQA datasets into training format for Qwen3-30B-A3B

Datasets:
- CORD: Receipt/Invoice extraction
- CUAD: Contract clause extraction  
- DocVQA: General document Q&A

Output: JSONL files ready for Unsloth training
"""

import json
import random
from pathlib import Path
from datasets import load_from_disk, load_dataset
from tqdm import tqdm
from PIL import Image
import base64
import io

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input paths - update these to match your setup
BASE_PATH = Path(__file__).parent
DATASETS_PATH = BASE_PATH / "data" / "datasets"

# Output paths
OUTPUT_DIR = BASE_PATH / "data" / "training"
TRAIN_FILE = OUTPUT_DIR / "train.jsonl"
EVAL_FILE = OUTPUT_DIR / "eval.jsonl"

# Training config
TRAIN_SPLIT = 0.9
MAX_EXAMPLES_PER_DATASET = 5000  # Limit per dataset to balance
SEED = 42

# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

SYSTEM_PROMPT = """You are an expert document analyst. Analyze the provided document image and extract structured information.

For invoices/receipts: Extract vendor, date, items, amounts, totals.
For contracts: Extract parties, terms, obligations, key clauses.
For general documents: Provide summary, key points, and relevant entities.

Always respond with:
1. **Document Type**: What kind of document this is
2. **Extracted Data**: JSON with structured fields
3. **Summary**: Brief natural language explanation"""

INVOICE_SYSTEM_PROMPT = """You are an expert invoice and receipt analyst. Extract structured data from the document image.

Respond with JSON containing:
- vendor: Company/store name
- date: Document date
- items: List of line items with description, quantity, price
- subtotal: Amount before tax
- tax: Tax amount
- total: Final amount
- currency: Currency used

Then provide a brief summary."""

CONTRACT_SYSTEM_PROMPT = """You are an expert legal document analyst. Extract key information from contracts and agreements.

Respond with JSON containing:
- document_type: Type of agreement
- parties: List of parties involved
- effective_date: When agreement starts
- key_terms: Important terms and conditions
- obligations: What each party must do
- termination: How the agreement can end

Then provide a brief summary of the contract."""

GENERAL_SYSTEM_PROMPT = """You are an expert document analyst. Answer questions about the document accurately and concisely.

Provide direct answers based on what you can see in the document. If the information is not visible, say so."""

# =============================================================================
# DATA CONVERSION FUNCTIONS
# =============================================================================

def convert_cord_example(example) -> dict:
    """Convert CORD receipt example to training format."""
    try:
        # Extract ground truth from CORD format
        ground_truth = example.get("ground_truth", {})
        
        # Parse the ground truth (CORD stores as JSON string sometimes)
        if isinstance(ground_truth, str):
            try:
                ground_truth = json.loads(ground_truth)
            except:
                ground_truth = {}
        
        # Build extracted data
        extracted = {
            "vendor": None,
            "date": None,
            "items": [],
            "subtotal": None,
            "tax": None,
            "total": None,
        }
        
        # Extract from CORD structure
        gt = ground_truth.get("gt_parse", ground_truth)
        
        # Get menu items
        menu = gt.get("menu", [])
        for item in menu:
            if isinstance(item, dict):
                extracted["items"].append({
                    "name": item.get("nm", ""),
                    "count": item.get("cnt", 1),
                    "price": item.get("price", "")
                })
        
        # Get totals
        total_info = gt.get("total", {})
        if isinstance(total_info, dict):
            extracted["total"] = total_info.get("total_price")
            extracted["tax"] = total_info.get("tax_price")
        
        sub_total = gt.get("sub_total", {})
        if isinstance(sub_total, dict):
            extracted["subtotal"] = sub_total.get("subtotal_price")
        
        # Build response
        response = f"""**Document Type**: Receipt/Invoice

**Extracted Data**:
```json
{json.dumps(extracted, indent=2)}
```

**Summary**: This is a receipt containing {len(extracted['items'])} item(s)."""
        
        if extracted["total"]:
            response += f" The total amount is {extracted['total']}."
        
        return {
            "messages": [
                {"role": "system", "content": INVOICE_SYSTEM_PROMPT},
                {"role": "user", "content": "Extract all information from this receipt image."},
                {"role": "assistant", "content": response}
            ],
            "image": example.get("image"),
            "_meta": {
                "source": "cord",
                "type": "invoice"
            }
        }
        
    except Exception as e:
        return None


def convert_cuad_example(example) -> dict:
    """Convert CUAD contract example to training format."""
    try:
        clause = example.get("clause", "").strip()
        label = example.get("label", "").strip()

        # Skip if no clause or label
        if not clause or not label:
            return None

        # Build response
        response = f"""**Document Type**: Legal Contract

**Clause Type**: {label}

**Extracted Clause**:
{clause}

**Summary**: This is a "{label}" clause from a commercial legal contract. It defines specific terms and obligations related to {label.lower()} provisions."""

        return {
            "messages": [
                {"role": "system", "content": CONTRACT_SYSTEM_PROMPT},
                {"role": "user", "content": f"Identify and analyze the following contract clause:\n\n{clause[:2000]}"},
                {"role": "assistant", "content": response}
            ],
            "_meta": {
                "source": "cuad",
                "type": "contract"
            }
        }

    except Exception as e:
        return None


def convert_docvqa_example(example) -> dict:
    """Convert DocVQA example to training format."""
    try:
        # Handle different DocVQA formats
        question = example.get("question") or example.get("questions", [""])[0]
        answers = example.get("answers", [])
        
        if isinstance(answers, dict):
            answer = answers.get("text", [""])[0] if answers.get("text") else ""
        elif isinstance(answers, list) and len(answers) > 0:
            answer = answers[0] if isinstance(answers[0], str) else str(answers[0])
        else:
            answer = str(answers) if answers else ""
        
        # Skip if no answer
        if not answer:
            return None
        
        response = f"""**Answer**: {answer}

This information was extracted directly from the document based on the question asked."""
        
        return {
            "messages": [
                {"role": "system", "content": GENERAL_SYSTEM_PROMPT},
                {"role": "user", "content": f"Looking at this document, answer the following question:\n\n{question}"},
                {"role": "assistant", "content": response}
            ],
            "image": example.get("image"),
            "_meta": {
                "source": "docvqa",
                "type": "general"
            }
        }
        
    except Exception as e:
        return None


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def load_and_convert_cord():
    """Load and convert CORD dataset."""
    print("\n[CORD] Loading receipt dataset...")
    
    examples = []
    
    try:
        # Try loading from disk first
        cord_path = DATASETS_PATH / "cord"
        if cord_path.exists():
            dataset = load_from_disk(str(cord_path))
        else:
            # Try HuggingFace
            dataset = load_dataset("naver-clova-ix/cord-v2")
        
        # Process train split
        split_name = "train" if "train" in dataset else list(dataset.keys())[0]
        data = dataset[split_name]
        
        print(f"[CORD] Processing {len(data)} examples...")
        
        for example in tqdm(data, desc="CORD"):
            converted = convert_cord_example(example)
            if converted:
                examples.append(converted)
            
            if len(examples) >= MAX_EXAMPLES_PER_DATASET:
                break
        
        print(f"[CORD] Converted {len(examples)} examples")
        
    except Exception as e:
        print(f"[CORD] Error: {e}")
    
    return examples


def load_and_convert_cuad():
    """Load and convert CUAD dataset."""
    print("\n[CUAD] Loading contract dataset...")
    
    examples = []
    
    try:
        # Try loading from disk first
        cuad_path = DATASETS_PATH / "cuad"
        if cuad_path.exists():
            dataset = load_from_disk(str(cuad_path))
        else:
            # Try HuggingFace
            dataset = load_dataset("theatticusproject/cuad-qa")
        
        # Process train split
        data = dataset["train"]
        
        print(f"[CUAD] Processing {len(data)} examples...")
        
        for example in tqdm(data, desc="CUAD"):
            converted = convert_cuad_example(example)
            if converted:
                examples.append(converted)
            
            if len(examples) >= MAX_EXAMPLES_PER_DATASET:
                break
        
        print(f"[CUAD] Converted {len(examples)} examples")
        
    except Exception as e:
        print(f"[CUAD] Error: {e}")
    
    return examples


def load_and_convert_docvqa():
    """Load and convert DocVQA dataset."""
    print("\n[DocVQA] Loading document QA dataset...")
    
    examples = []
    
    try:
        # Try loading from disk first
        docvqa_path = DATASETS_PATH / "docvqa"
        if docvqa_path.exists():
            dataset = load_from_disk(str(docvqa_path))
        else:
            # Try HuggingFace
            dataset = load_dataset("HuggingFaceM4/DocumentVQA")
        
        # Process train split
        split_name = "train" if "train" in dataset else list(dataset.keys())[0]
        data = dataset[split_name]
        
        print(f"[DocVQA] Processing {len(data)} examples...")
        
        for example in tqdm(data, desc="DocVQA"):
            converted = convert_docvqa_example(example)
            if converted:
                examples.append(converted)
            
            if len(examples) >= MAX_EXAMPLES_PER_DATASET:
                break
        
        print(f"[DocVQA] Converted {len(examples)} examples")
        
    except Exception as e:
        print(f"[DocVQA] Error: {e}")
    
    return examples


def save_training_data(examples: list):
    """Save examples to train/eval splits."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Shuffle
    random.seed(SEED)
    random.shuffle(examples)
    
    # Split
    split_idx = int(len(examples) * TRAIN_SPLIT)
    train_examples = examples[:split_idx]
    eval_examples = examples[split_idx:]
    
    # Save train
    print(f"\nSaving {len(train_examples)} training examples...")
    with open(TRAIN_FILE, "w") as f:
        for ex in train_examples:
            # Remove image data for text-only training (add back for VL training)
            ex_copy = {k: v for k, v in ex.items() if k != "image"}
            f.write(json.dumps(ex_copy) + "\n")
    
    # Save eval
    print(f"Saving {len(eval_examples)} eval examples...")
    with open(EVAL_FILE, "w") as f:
        for ex in eval_examples:
            ex_copy = {k: v for k, v in ex.items() if k != "image"}
            f.write(json.dumps(ex_copy) + "\n")
    
    # Stats
    stats = {
        "total": len(examples),
        "train": len(train_examples),
        "eval": len(eval_examples),
        "by_source": {},
        "by_type": {}
    }
    
    for ex in examples:
        source = ex.get("_meta", {}).get("source", "unknown")
        doc_type = ex.get("_meta", {}).get("type", "unknown")
        stats["by_source"][source] = stats["by_source"].get(source, 0) + 1
        stats["by_type"][doc_type] = stats["by_type"].get(doc_type, 0) + 1
    
    # Save stats
    with open(OUTPUT_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    return stats


def main():
    print("=" * 60)
    print("PREPARE TRAINING DATA FOR DOCUMENT INTELLIGENCE MOE")
    print("=" * 60)
    print(f"Base path: {BASE_PATH}")
    print(f"Datasets path: {DATASETS_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    
    all_examples = []
    
    # Load and convert each dataset
    cord_examples = load_and_convert_cord()
    all_examples.extend(cord_examples)
    
    cuad_examples = load_and_convert_cuad()
    all_examples.extend(cuad_examples)
    
    docvqa_examples = load_and_convert_docvqa()
    all_examples.extend(docvqa_examples)
    
    # Save
    if all_examples:
        stats = save_training_data(all_examples)
        
        print("\n" + "=" * 60)
        print("DATASET PREPARATION COMPLETE")
        print("=" * 60)
        print(f"Total examples: {stats['total']}")
        print(f"Train: {stats['train']}")
        print(f"Eval: {stats['eval']}")
        print(f"\nBy source:")
        for source, count in stats["by_source"].items():
            print(f"  {source}: {count}")
        print(f"\nBy type:")
        for doc_type, count in stats["by_type"].items():
            print(f"  {doc_type}: {count}")
        print(f"\nFiles saved:")
        print(f"  {TRAIN_FILE}")
        print(f"  {EVAL_FILE}")
    else:
        print("\n❌ No examples converted. Check dataset paths.")


if __name__ == "__main__":
    main()
