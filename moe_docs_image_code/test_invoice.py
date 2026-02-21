"""
Invoice extraction test — tests both LoRA adapters and Ollama deployment.
Usage:
    python3 test_invoice.py                          # uses default PDF
    python3 test_invoice.py /path/to/invoice.pdf    # custom PDF
    python3 test_invoice.py --lora-only             # skip Ollama
    python3 test_invoice.py --ollama-only           # skip LoRA (faster)
"""

import sys
import json
import torch
import requests
import pdfplumber
from pathlib import Path
from unsloth import FastLanguageModel

# =============================================================================
# CONFIG
# =============================================================================

BASE_PATH = Path(__file__).parent
MODEL_PATH = BASE_PATH / "output" / "lora_adapters"
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "doc-intel"

DEFAULT_PDF = Path("/home/irfana/Documents/Data/invoices/archive (2)/2024/ca/20240909_Hilton.pdf")

SYSTEM_PROMPT = """You are an expert invoice and receipt analyst. Extract all information from the document and respond with valid JSON only — no explanation, no markdown, just the JSON object.

The JSON should contain:
- vendor: { name, address }
- invoice_number: folio or invoice number (string)
- date: invoice date (YYYY-MM-DD)
- guest: guest or customer name
- check_in: check-in or service start date (YYYY-MM-DD, null if not applicable)
- check_out: check-out or service end date (YYYY-MM-DD, null if not applicable)
- items: [ { description, date, amount } ]
- subtotal: numeric amount before tax
- taxes: [ { name, amount } ]
- total: final total as numeric
- currency: ISO currency code (e.g. USD, CAD)
- payment_method: how it was paid (null if not shown)"""

# =============================================================================
# PDF EXTRACTION
# =============================================================================

def extract_pdf_text(pdf_path: Path) -> str:
    """Extract all text from a PDF using pdfplumber."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                pages.append(f"[Page {i+1}]\n{text.strip()}")
    return "\n\n".join(pages)


def parse_json_from_response(response: str) -> dict | None:
    """Try to extract a JSON object from the model response."""
    # Strip markdown code fences if present
    response = response.strip()
    if response.startswith("```"):
        lines = response.split("\n")
        response = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    start = response.find("{")
    end = response.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(response[start:end])
        except json.JSONDecodeError:
            pass
    return None


# =============================================================================
# TEST VIA LORA ADAPTERS
# =============================================================================

def test_lora(pdf_text: str) -> str:
    """Test using LoRA adapters loaded directly via Unsloth."""
    print("\n" + "=" * 60)
    print("TEST 1: LoRA Adapters (direct — bypasses Ollama)")
    print("=" * 60)

    if not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}")
        return ""

    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(MODEL_PATH),
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    # Qwen1.5-MoE-A2.7B uses ~7GB for weights, leaving ~8GB for KV cache.
    # At 2048 max_seq_length, can handle full invoice documents comfortably.
    MAX_PDF_CHARS = 6000
    if len(pdf_text) > MAX_PDF_CHARS:
        print(f"(Truncating PDF text from {len(pdf_text)} to {MAX_PDF_CHARS} chars to fit in VRAM)")
        pdf_text = pdf_text[:MAX_PDF_CHARS]

    # Free any cached memory before generation
    torch.cuda.empty_cache()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Extract all information from this invoice:\n\n{pdf_text}"},
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[1]
    print(f"Input tokens: {input_length}")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    print("\nRaw response:")
    print(response)

    parsed = parse_json_from_response(response)
    if parsed:
        print("\nParsed JSON:")
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
    else:
        print("\n(Response is not valid JSON)")

    return response


# =============================================================================
# TEST VIA OLLAMA
# =============================================================================

def test_ollama(pdf_text: str) -> str:
    """Test using Ollama with the deployed GGUF model."""
    print("\n" + "=" * 60)
    print("TEST 2: Ollama GGUF Deployment")
    print("=" * 60)

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract all information from this invoice:\n\n{pdf_text}"},
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "top_k": 40,
        },
    }

    try:
        print(f"Querying {OLLAMA_MODEL} via Ollama...")
        resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
        resp.raise_for_status()
        full = resp.json()
        response = full["message"]["content"]

        # Debug: show stop reason and token count
        print(f"Stop reason: {full.get('done_reason', 'unknown')}  |  "
              f"Tokens generated: {full.get('eval_count', '?')}")
        print("\nRaw response:")
        print(repr(response) if not response.strip() else response)

        parsed = parse_json_from_response(response)
        if parsed:
            print("\nParsed JSON:")
            print(json.dumps(parsed, indent=2, ensure_ascii=False))
        else:
            print("\n(Response is not valid JSON)")

        return response

    except requests.exceptions.ConnectionError:
        print("Ollama is not running. Start it with: ollama serve")
        print("Then reload the model: ollama create doc-intel -f output/gguf/doc-intel_gguf/Modelfile")
        return ""
    except Exception as e:
        print(f"Ollama error: {e}")
        return ""


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = sys.argv[1:]
    lora_only = "--lora-only" in args
    ollama_only = "--ollama-only" in args
    args = [a for a in args if not a.startswith("--")]

    pdf_path = Path(args[0]) if args else DEFAULT_PDF

    print("=" * 60)
    print("INVOICE EXTRACTION TEST")
    print(f"PDF: {pdf_path.name}")
    print("=" * 60)

    if not pdf_path.exists():
        print(f"\nPDF not found: {pdf_path}")
        return

    # Extract text from PDF
    print("\nExtracting text from PDF...")
    pdf_text = extract_pdf_text(pdf_path)
    print(f"Extracted {len(pdf_text)} characters across the document.\n")
    print("--- PDF TEXT PREVIEW (first 800 chars) ---")
    print(pdf_text[:800] + ("..." if len(pdf_text) > 800 else ""))
    print("--- END PREVIEW ---")

    # Run tests
    if not ollama_only:
        test_lora(pdf_text)

    if not lora_only:
        test_ollama(pdf_text)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
