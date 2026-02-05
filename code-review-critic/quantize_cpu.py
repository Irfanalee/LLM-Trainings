"""
Quantize the fine-tuned model to GGUF format for CPU deployment.

Uses llama.cpp conversion (CPU-based) to avoid VRAM limitations.
Creates a Q4_K_M quantized model (~4-5GB) that runs on CPU with 16GB RAM.

Usage:
    python quantize.py
"""

import os
import subprocess
import sys
from pathlib import Path

# Configuration
MODEL_PATH = "./output/merged_model_v2"
OUTPUT_DIR = "./output/gguf"
LLAMA_CPP_DIR = "./llama.cpp"

# Quantization method - Q4_K_M is good balance of size/quality
# Options: Q4_K_M, Q5_K_M, Q8_0, F16
QUANT_METHOD = "Q4_K_M"


def run_cmd(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{description}...")
    print(f"$ {cmd}\n")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Error: {description} failed!")
        sys.exit(1)


def main():
    print("=" * 60)
    print("GGUF QUANTIZATION (CPU-based)")
    print("=" * 60)
    print(f"Input: {MODEL_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Method: {QUANT_METHOD}")
    print("\nThis uses llama.cpp and runs on CPU - no GPU memory needed.")
    print()

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check if llama.cpp exists
    if not Path(LLAMA_CPP_DIR).exists():
        print("llama.cpp not found. Cloning...")
        run_cmd(
            "git clone https://github.com/ggerganov/llama.cpp.git",
            "Cloning llama.cpp"
        )

    # Build llama.cpp if needed (uses CMake now)
    quantize_bin = Path(LLAMA_CPP_DIR) / "build" / "bin" / "llama-quantize"
    if not quantize_bin.exists():
        print("\nBuilding llama.cpp with CMake (this may take a few minutes)...")
        run_cmd(
            f"cd {LLAMA_CPP_DIR} && cmake -B build && cmake --build build --config Release -j$(nproc)",
            "Building llama.cpp"
        )

    # Install required Python packages for conversion
    run_cmd(
        "pip install -q sentencepiece protobuf",
        "Installing conversion dependencies"
    )

    # Step 1: Convert HF model to GGUF (F16)
    f16_path = f"{OUTPUT_DIR}/model-f16.gguf"
    print("\n" + "=" * 60)
    print("Step 1: Convert HuggingFace model to GGUF (F16)")
    print("=" * 60)
    print("This reads model files directly - no GPU needed.")

    run_cmd(
        f"python {LLAMA_CPP_DIR}/convert_hf_to_gguf.py {MODEL_PATH} --outfile {f16_path} --outtype f16",
        "Converting to GGUF F16"
    )

    # Step 2: Quantize to Q4_K_M
    q4_path = f"{OUTPUT_DIR}/code-review-critic-{QUANT_METHOD.lower()}.gguf"
    print("\n" + "=" * 60)
    print(f"Step 2: Quantize to {QUANT_METHOD}")
    print("=" * 60)

    run_cmd(
        f"{LLAMA_CPP_DIR}/build/bin/llama-quantize {f16_path} {q4_path} {QUANT_METHOD}",
        f"Quantizing to {QUANT_METHOD}"
    )

    # Clean up F16 file (it's ~15GB)
    print(f"\nRemoving intermediate F16 file to save space...")
    if Path(f16_path).exists():
        os.remove(f16_path)
        print(f"Removed {f16_path}")

    # Show file size
    if Path(q4_path).exists():
        q4_size = Path(q4_path).stat().st_size / (1024**3)
        print(f"\nQuantized model: {q4_path}")
        print(f"Size: {q4_size:.2f} GB")

    print("\n" + "=" * 60)
    print("QUANTIZATION COMPLETE")
    print("=" * 60)

    # Print Ollama setup instructions
    print(f"""
OLLAMA SETUP
============

1. Create a Modelfile:

cat > Modelfile << 'EOF'
FROM {q4_path}

TEMPLATE \"\"\"<|im_start|>system
{{{{ .System }}}}<|im_end|>
<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
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
