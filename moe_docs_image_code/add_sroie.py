"""
Add CORD validation + test splits to existing training files.
Appends to train.jsonl and eval.jsonl without reprocessing other datasets.
"""

import json
import random
from pathlib import Path
from datasets import load_from_disk
from tqdm import tqdm

BASE_PATH    = Path(__file__).parent
DATASETS_PATH = BASE_PATH / "data" / "datasets"
OUTPUT_DIR   = BASE_PATH / "data" / "training"
TRAIN_FILE   = OUTPUT_DIR / "train.jsonl"
EVAL_FILE    = OUTPUT_DIR / "eval.jsonl"

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


def convert_cord_example(example) -> dict:
    try:
        ground_truth = example.get("ground_truth", {})
        if isinstance(ground_truth, str):
            try:
                ground_truth = json.loads(ground_truth)
            except Exception:
                ground_truth = {}

        extracted = {"vendor": None, "date": None, "items": [], "subtotal": None, "tax": None, "total": None}
        gt = ground_truth.get("gt_parse", ground_truth)

        menu = gt.get("menu", [])
        for item in menu:
            if isinstance(item, dict):
                extracted["items"].append({
                    "name": item.get("nm", ""),
                    "count": item.get("cnt", 1),
                    "price": item.get("price", "")
                })

        total_info = gt.get("total", {})
        if isinstance(total_info, dict):
            extracted["total"] = total_info.get("total_price")
            extracted["tax"]   = total_info.get("tax_price")

        sub_total = gt.get("sub_total", {})
        if isinstance(sub_total, dict):
            extracted["subtotal"] = sub_total.get("subtotal_price")

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
                {"role": "system",    "content": INVOICE_SYSTEM_PROMPT},
                {"role": "user",      "content": "Extract all information from this receipt image."},
                {"role": "assistant", "content": response}
            ],
            "_meta": {"source": "cord", "type": "invoice"}
        }
    except Exception:
        return None


def main():
    cord_path = DATASETS_PATH / "cord"
    print(f"Loading CORD from {cord_path} ...")
    dataset = load_from_disk(str(cord_path), keep_in_memory=False)

    examples = []
    for split in ["validation", "test"]:
        if split not in dataset:
            print(f"  Split '{split}' not found, skipping.")
            continue
        data = dataset[split]
        print(f"  Processing {split} ({len(data)} examples)...")
        for example in tqdm(data, desc=f"CORD/{split}"):
            converted = convert_cord_example(example)
            if converted:
                examples.append(converted)

    print(f"\nConverted {len(examples)} examples")

    # Split and append
    random.seed(SEED)
    random.shuffle(examples)
    split_idx      = int(len(examples) * TRAIN_SPLIT)
    train_examples = examples[:split_idx]
    eval_examples  = examples[split_idx:]

    with open(TRAIN_FILE, "a") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")

    with open(EVAL_FILE, "a") as f:
        for ex in eval_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Appended {len(train_examples)} to train.jsonl")
    print(f"Appended {len(eval_examples)} to eval.jsonl")
    print("Done.")


if __name__ == "__main__":
    main()
