"""
Model Download Script
Downloads and caches the required models for Document Intelligence Agent

Models:
1. Florence-2-large - Microsoft's vision-language model for document understanding
2. Qwen2.5-7B-Instruct - Alibaba's instruction-tuned LLM for reasoning

Both models fit comfortably in 16GB VRAM when using 4-bit quantization
"""

import os
import torch
from pathlib import Path

# Disable tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def check_disk_space(required_gb=30):
    """Check if enough disk space is available"""
    import shutil
    total, used, free = shutil.disk_usage("/")
    free_gb = free / (1024**3)
    print(f"Free disk space: {free_gb:.1f} GB")
    if free_gb < required_gb:
        print(f"⚠️  Warning: Less than {required_gb} GB free. Models may not download properly.")
        return False
    return True


def download_florence2():
    """Download Florence-2-large for document understanding"""
    print("\n" + "=" * 50)
    print("Downloading Florence-2-large")
    print("=" * 50)
    
    from transformers import AutoProcessor, AutoModelForCausalLM
    
    model_name = "microsoft/Florence-2-large"
    
    print(f"\nDownloading from: {model_name}")
    print("This model is ~1.5GB and excels at:")
    print("  - Document OCR and text extraction")
    print("  - Visual question answering")
    print("  - Object detection and region understanding")
    print("  - Dense captioning")
    
    try:
        # Download processor (tokenizer + image processor)
        print("\n[1/2] Downloading processor...")
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        print("  ✓ Processor downloaded")
        
        # Download model
        print("\n[2/2] Downloading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        print("  ✓ Model downloaded")
        
        # Quick test
        print("\n[Test] Verifying model loads correctly...")
        model = model.to("cuda")
        print(f"  ✓ Model loaded to GPU")
        print(f"  ✓ Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        
        # Cleanup to free memory
        del model
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error downloading Florence-2: {e}")
        return False


def download_qwen():
    """Download Qwen2.5-7B-Instruct for reasoning"""
    print("\n" + "=" * 50)
    print("Downloading Qwen2.5-7B-Instruct (4-bit quantized)")
    print("=" * 50)
    
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    
    print(f"\nDownloading from: {model_name}")
    print("This model excels at:")
    print("  - Complex reasoning and analysis")
    print("  - Following instructions precisely")
    print("  - Structured output generation")
    print("  - Multi-turn conversations")
    
    try:
        # Download tokenizer
        print("\n[1/2] Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("  ✓ Tokenizer downloaded")
        
        # Configure 4-bit quantization for memory efficiency
        print("\n[2/2] Downloading model with 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        print("  ✓ Model downloaded and quantized")
        
        # Check memory usage
        allocated = torch.cuda.memory_allocated(0) / 1e9
        print(f"\n  GPU Memory used: {allocated:.2f} GB")
        
        # Cleanup
        del model, tokenizer
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error downloading Qwen: {e}")
        return False


def download_alternative_small():
    """Download smaller alternative models if main ones fail"""
    print("\n" + "=" * 50)
    print("Downloading Alternative: Phi-3-mini (smaller footprint)")
    print("=" * 50)
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    
    try:
        print("\n[1/2] Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print("  ✓ Tokenizer downloaded")
        
        print("\n[2/2] Downloading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        print("  ✓ Model downloaded")
        
        del model, tokenizer
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    print("=" * 50)
    print("Document Intelligence Agent - Model Download")
    print("=" * 50)
    
    # Check prerequisites
    print("\n[Prerequisite Check]")
    
    # Check disk space
    if not check_disk_space(30):
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("✗ CUDA not available. Please check your GPU setup.")
        return
    print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    
    # Download models
    results = {}
    
    # Florence-2 for vision
    results['florence2'] = download_florence2()
    
    # Qwen for reasoning
    results['qwen'] = download_qwen()
    
    # Summary
    print("\n" + "=" * 50)
    print("Download Summary")
    print("=" * 50)
    
    for model, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {model}")
    
    if all(results.values()):
        print("\n✓ All models downloaded successfully!")
        print("\nNext steps:")
        print("  1. Run: python src/inference_demo.py")
        print("  2. Or launch the Gradio UI: python demos/gradio_app.py")
    else:
        print("\n⚠️  Some models failed to download.")
        print("Check your internet connection and disk space.")


if __name__ == "__main__":
    main()
