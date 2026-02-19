"""
Export Fine-tuned Model to GGUF format for Ollama / llama.cpp
"""

import os
from pathlib import Path
from unsloth import FastLanguageModel

BASE_PATH  = Path(__file__).parent
MODEL_PATH = BASE_PATH / "output" / "lora_adapters"
GGUF_PATH  = BASE_PATH / "output" / "gguf"
MAX_SEQ_LENGTH = 1536

# Quantization options (pick one):
#   "q4_k_m"  — best balance of size and quality (recommended)
#   "q5_k_m"  — better quality, larger file
#   "q8_0"    — near-lossless, largest file
#   "f16"     — full precision, very large
QUANTIZATION = "q4_k_m"


def main():
    print("=" * 60)
    print("EXPORT MODEL TO GGUF")
    print("=" * 60)

    if not MODEL_PATH.exists():
        print(f"Model not found at: {MODEL_PATH}")
        print("Run training first: python3 train_moe.py")
        return

    GGUF_PATH.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from: {MODEL_PATH}")
    print(f"Output:             {GGUF_PATH}")
    print(f"Quantization:       {QUANTIZATION}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(MODEL_PATH),
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    print("\nExporting to GGUF...")
    model.save_pretrained_gguf(
        str(GGUF_PATH / "doc-intel"),
        tokenizer,
        quantization_method=QUANTIZATION,
    )

    # Unsloth saves into a subdirectory e.g. doc-intel_gguf/
    gguf_file = next(GGUF_PATH.rglob("*.gguf"), None)
    modelfile  = next(GGUF_PATH.rglob("Modelfile"), None)

    print(f"\nGGUF file:  {gguf_file}")
    print(f"Modelfile:  {modelfile}")
    print("\nTo deploy with Ollama:")
    if modelfile:
        print(f"  ollama create doc-intel -f {modelfile}")
    print(f"  ollama run doc-intel")


if __name__ == "__main__":
    main()
