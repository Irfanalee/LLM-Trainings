"""
Add SROIE receipt data to existing training files.
Appends to train.jsonl and eval.jsonl without reprocessing other datasets.
"""

import json
import random
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

BASE_PATH = Path(__file__).parent
OUTPUT_DIR = BASE_PATH / "data" / "training"
TRAIN_FILE = OUTPUT_DIR / "train.jsonl"
EVAL_FILE  = OUTPUT_DIR / "eval.jsonl"

TRAIN_SPLIT = 0.9
SEED = 42

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


def convert_sroie_example(example) -> dict:
    company = (example.get("company") or "").strip()
    date    = (example.get("date")    or "").strip()
    address = (example.get("address") or "").strip()
    total   = (example.get("total")   or "").strip()

    if not any([company, date, total]):
        return None

    extracted = {"vendor": company, "date": date, "address": address, "total": total}

    response = f"""**Document Type**: Receipt/Invoice

**Extracted Data**:
```json
{json.dumps(extracted, indent=2)}
```

**Summary**: Receipt from {company or 'unknown vendor'} dated {date or 'unknown date'}. Total: {total or 'unknown'}."""

    return {
        "messages": [
            {"role": "system",    "content": INVOICE_SYSTEM_PROMPT},
            {"role": "user",      "content": "Extract all information from this receipt image."},
            {"role": "assistant", "content": response}
        ],
        "_meta": {"source": "sroie", "type": "invoice"}
    }


def main():
    print("Loading SROIE from HuggingFace...")
    dataset = load_dataset("jinhybr/OCR-SROIE-English", split="train")
    print(f"Processing {len(dataset)} examples...")

    examples = []
    for example in tqdm(dataset, desc="SROIE"):
        converted = convert_sroie_example(example)
        if converted:
            examples.append(converted)

    print(f"Converted {len(examples)} examples")

    # Split
    random.seed(SEED)
    random.shuffle(examples)
    split_idx     = int(len(examples) * TRAIN_SPLIT)
    train_examples = examples[:split_idx]
    eval_examples  = examples[split_idx:]

    # Append to existing files
    with open(TRAIN_FILE, "a") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")

    with open(EVAL_FILE, "a") as f:
        for ex in eval_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\nAppended {len(train_examples)} to train.jsonl")
    print(f"Appended {len(eval_examples)} to eval.jsonl")
    print("Done.")


if __name__ == "__main__":
    main()
