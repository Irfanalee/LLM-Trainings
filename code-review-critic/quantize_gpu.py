"""
Quantize the fine-tuned model to GGUF format using GPU (Unsloth).

WARNING: Requires ~15-16GB VRAM. May OOM on RTX A4000 16GB.
         Use quantize_cpu.py instead for 16GB GPUs.

Usage:
    python quantize_gpu.py
"""

import os
from unsloth import FastLanguageModel

# Configuration
MODEL_PATH = "./output/merged_model_v2"
OUTPUT_DIR = "./output/gguf"
MAX_SEQ_LENGTH = 1536

# Quantization method - q4_k_m is good balance of size/quality
# Options: q4_k_m, q5_k_m, q8_0, f16
QUANT_METHOD = "q4_k_m"


def main():
    print("=" * 60)
    print("GGUF QUANTIZATION (GPU-based)")
    print("=" * 60)
    print(f"Input: {MODEL_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Method: {QUANT_METHOD}")
    print()
    print("WARNING: This requires ~15-16GB VRAM.")
    print("         If you get OOM, use quantize_cpu.py instead.")
    print()

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load the merged model
    print("Loading model (this uses ~15GB VRAM)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=False,  # Load in full precision for quantization
    )
    print("Model loaded.\n")

    # Export to GGUF
    print(f"Quantizing to {QUANT_METHOD.upper()}...")
    print("This may take 10-20 minutes...\n")

    model.save_pretrained_gguf(
        OUTPUT_DIR,
        tokenizer,
        quantization_method=QUANT_METHOD,
    )

    print("\n" + "=" * 60)
    print("QUANTIZATION COMPLETE")
    print("=" * 60)
    print(f"\nOutput files in: {OUTPUT_DIR}/")
    print("\nThe GGUF file will be named: unsloth.Q4_K_M.gguf")
    print()

    # Print Ollama setup instructions
    print("=" * 60)
    print("OLLAMA SETUP")
    print("=" * 60)
    print("""
1. Create a Modelfile:

cat > Modelfile << 'EOF'
FROM ./output/gguf/unsloth.Q4_K_M.gguf

TEMPLATE \"\"\"<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
\"\"\"

SYSTEM "You are an expert code reviewer. Analyze the provided Python code and give constructive, specific feedback. Focus on bugs, potential issues, code quality, and improvements. Be direct and actionable."

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "<|im_end|>"
EOF

2. Create the model in Ollama:

ollama create code-review-critic -f Modelfile

3. Test it:

ollama run code-review-critic "Review this Python code:

def get_user(user_id):
    user = db.query(User).filter(id=user_id).first()
    return user.name
"
""")


if __name__ == "__main__":
    main()
