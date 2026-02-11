#!/usr/bin/env python3
"""
CPU Performance Benchmark for Code Review Critic Model
Measures tokens per second (t/s) for inference on CPU
"""

import time
import sys
from pathlib import Path

try:
    from llama_cpp import Llama
except ImportError:
    print("ERROR: llama-cpp-python not installed")
    print("Install with: pip install llama-cpp-python")
    sys.exit(1)


# Model and settings
MODEL_PATH = Path(__file__).parent / "output/gguf/code-review-critic-q4_k_m.gguf"
MAX_TOKENS = 256
NUM_THREADS = None  # None = use all available CPU cores
PROMPT_TOKENS = 50  # Approximate prompt size for measurements


def benchmark_prompt_processing():
    """Benchmark prompt tokenization speed (tokens/s)"""
    print("\n" + "="*70)
    print("PROMPT PROCESSING BENCHMARK")
    print("="*70)
    
    prompt = """<|im_start|>system
You are an expert code reviewer. Analyze the provided Python code and give constructive, specific feedback. Focus on bugs, potential issues, code quality, and improvements. Be direct and actionable.<|im_end|>
<|im_start|>user
Review this Python code from `example.py`:

```python
def get_user(user_id):
    user = db.query(User).filter(id=user_id).first()
    return user.name
```<|im_end|>
<|im_start|>assistant
"""
    
    llm = Llama(
        model_path=str(MODEL_PATH),
        n_threads=NUM_THREADS,
        n_gpu_layers=0,  # Force CPU only
        verbose=False
    )
    
    # Warm-up
    _ = llm(prompt, max_tokens=10)
    
    # Benchmark prompt processing
    print(f"\nTesting prompt processing with {len(prompt)} characters...")
    start = time.perf_counter()
    
    response = llm(
        prompt,
        max_tokens=MAX_TOKENS,
        temperature=0.7,
        top_p=0.9,
    )
    
    end = time.perf_counter()
    elapsed = end - start
    
    # Parse results
    total_tokens = response["usage"]["total_tokens"]
    prompt_tokens = response["usage"]["prompt_tokens"]
    completion_tokens = response["usage"]["completion_tokens"]
    
    # Calculate tokens per second
    prompt_tps = prompt_tokens / elapsed if elapsed > 0 else 0
    completion_tps = completion_tokens / elapsed if elapsed > 0 else 0
    total_tps = total_tokens / elapsed if elapsed > 0 else 0
    
    print(f"\nResults:")
    print(f"  Total time: {elapsed:.2f} seconds")
    print(f"  Prompt tokens: {prompt_tokens}")
    print(f"  Completion tokens: {completion_tokens}")
    print(f"  Total tokens: {total_tokens}")
    print(f"\nThroughput:")
    print(f"  Prompt processing: {prompt_tps:.2f} tokens/s")
    print(f"  Token generation: {completion_tps:.2f} tokens/s")
    print(f"  Overall: {total_tps:.2f} tokens/s")
    
    print(f"\nGenerated review:")
    print("-" * 70)
    print(response["choices"][0]["text"].strip())
    print("-" * 70)
    
    return {
        "elapsed": elapsed,
        "prompt_tps": prompt_tps,
        "completion_tps": completion_tps,
        "total_tps": total_tps,
        "total_tokens": total_tokens
    }


def benchmark_generation():
    """Benchmark token generation speed (tokens/s)"""
    print("\n" + "="*70)
    print("TOKEN GENERATION BENCHMARK")
    print("="*70)
    
    prompts = [
        """<|im_start|>system
You are an expert code reviewer.<|im_end|>
<|im_start|>user
What is wrong with this code?

```python
def divide(a, b):
    return a / b
```<|im_end|>
<|im_start|>assistant
""",
        """<|im_start|>system
You are a code quality expert.<|im_end|>
<|im_start|>user
Review this function:

```python
def process_data(data):
    result = []
    for i in range(len(data)):
        result.append(data[i] * 2)
    return result
```<|im_end|>
<|im_start|>assistant
""",
        """<|im_start|>system
You provide code feedback.<|im_end|>
<|im_start|>user
Is this code correct?

```python
def get_value(d, key):
    return d[key]
```<|im_end|>
<|im_start|>assistant
"""
    ]
    
    llm = Llama(
        model_path=str(MODEL_PATH),
        n_threads=NUM_THREADS,
        n_gpu_layers=0,  # Force CPU only
        verbose=False
    )
    
    # Warm-up
    _ = llm(prompts[0], max_tokens=10)
    
    results = []
    for i, prompt in enumerate(prompts, 1):
        print(f"\n[Test {i}/{len(prompts)}]")
        
        start = time.perf_counter()
        response = llm(
            prompt,
            max_tokens=MAX_TOKENS,
            temperature=0.7,
            top_p=0.9,
        )
        end = time.perf_counter()
        
        elapsed = end - start
        total_tokens = response["usage"]["total_tokens"]
        completion_tokens = response["usage"]["completion_tokens"]
        tps = total_tokens / elapsed if elapsed > 0 else 0
        
        results.append(tps)
        
        print(f"  Time: {elapsed:.2f}s | Tokens: {total_tokens} | Speed: {tps:.2f} t/s")
    
    avg_tps = sum(results) / len(results) if results else 0
    min_tps = min(results) if results else 0
    max_tps = max(results) if results else 0
    
    print(f"\nSummary:")
    print(f"  Average: {avg_tps:.2f} tokens/s")
    print(f"  Min: {min_tps:.2f} tokens/s")
    print(f"  Max: {max_tps:.2f} tokens/s")
    
    return {
        "avg_tps": avg_tps,
        "min_tps": min_tps,
        "max_tps": max_tps,
        "results": results
    }


def main():
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print(f"Available files in output/gguf/:")
        gguf_dir = MODEL_PATH.parent
        if gguf_dir.exists():
            for f in gguf_dir.iterdir():
                print(f"  - {f.name}")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("CODE REVIEW CRITIC - CPU BENCHMARK")
    print("="*70)
    print(f"\nModel: {MODEL_PATH.name}")
    print(f"Max tokens per response: {MAX_TOKENS}")
    print(f"CPU threads: {NUM_THREADS if NUM_THREADS else 'all available'}")
    
    try:
        # Run benchmarks
        prompt_results = benchmark_prompt_processing()
        gen_results = benchmark_generation()
        
        # Summary
        print("\n" + "="*70)
        print("FINAL SUMMARY")
        print("="*70)
        print(f"\nPrompt Processing Speed: {prompt_results['prompt_tps']:.2f} tokens/s")
        print(f"Token Generation Speed (avg): {gen_results['avg_tps']:.2f} tokens/s")
        print(f"Overall Throughput: {prompt_results['total_tps']:.2f} tokens/s")
        print("\n✓ Benchmark complete!")
        
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        print("\nTroubleshooting:")
        print("1. Install llama-cpp-python: pip install llama-cpp-python")
        print("2. Ensure the GGUF model file exists at:", MODEL_PATH)
        print("3. Check that your system has enough RAM (8GB+ recommended)")
        sys.exit(1)


if __name__ == "__main__":
    main()
