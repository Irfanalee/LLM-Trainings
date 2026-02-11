#!/usr/bin/env python3
"""
CPU Performance Benchmark for Code Review Critic Model (using Ollama/Docker)
Measures tokens per second (t/s) for inference on CPU
"""

import subprocess
import json
import time
import sys
from pathlib import Path


GGUF_MODEL = "code-review-critic-q4_k_m.gguf"
GGUF_PATH = Path(__file__).parent / "output/gguf" / GGUF_MODEL


def check_ollama():
    """Check if Ollama is installed and running"""
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        print(f"✓ Ollama found: {result.stdout.strip()}")
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("✗ Ollama not found. Install from https://ollama.ai")
        return False


def check_model_loaded():
    """Check if the model is loaded in Ollama"""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if "code-review-critic" in result.stdout:
            print("✓ Model is loaded in Ollama")
            return True
        else:
            print("✗ Model not loaded. Run: ollama create code-review-critic -f Modelfile")
            return False
    except Exception as e:
        print(f"✗ Error checking model: {e}")
        return False


def benchmark_cpu_inference():
    """Benchmark CPU inference using Ollama"""
    
    test_prompts = [
        {
            "name": "Simple bug detection",
            "prompt": """Review this Python code for bugs:

```python
def get_user(user_id):
    user = db.query(User).filter(id=user_id).first()
    return user.name
```"""
        },
        {
            "name": "Error handling check",
            "prompt": """Review this code:

```python
def fetch_data(url):
    response = requests.get(url)
    return response.json()
```"""
        },
        {
            "name": "Data access check",
            "prompt": """Review this function:

```python
def parse_config(config_file):
    with open(config_file) as f:
        data = json.load(f)
    return data["settings"]["database"]["host"]
```"""
        },
    ]
    
    print("\n" + "="*70)
    print("CPU BENCHMARK: Code Review Critic Model")
    print("="*70)
    
    results = []
    
    for i, test in enumerate(test_prompts, 1):
        print(f"\n[Test {i}] {test['name']}")
        print("-" * 70)
        
        try:
            # Create the request with timing
            start = time.perf_counter()
            
            result = subprocess.run(
                ["ollama", "run", "code-review-critic", test["prompt"]],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            end = time.perf_counter()
            elapsed = end - start
            
            if result.returncode == 0:
                response = result.stdout.strip()
                
                # Rough token count (1 token ≈ 4 characters)
                response_tokens = len(response) // 4
                prompt_tokens = len(test["prompt"]) // 4
                total_tokens = response_tokens + prompt_tokens
                
                tps = total_tokens / elapsed if elapsed > 0 else 0
                
                print(f"Time: {elapsed:.2f}s")
                print(f"Est. tokens: {total_tokens} ({prompt_tokens} prompt + {response_tokens} response)")
                print(f"Throughput: {tps:.2f} tokens/s")
                print(f"\nResponse preview:")
                print(response[:200] + ("..." if len(response) > 200 else ""))
                
                results.append({
                    "name": test["name"],
                    "elapsed": elapsed,
                    "tokens": total_tokens,
                    "tps": tps
                })
            else:
                print(f"✗ Error: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print("✗ Timeout (>5 minutes)")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    # Summary
    if results:
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        
        avg_tps = sum(r["tps"] for r in results) / len(results)
        min_tps = min(r["tps"] for r in results)
        max_tps = max(r["tps"] for r in results)
        
        for r in results:
            print(f"{r['name']:30} {r['tps']:8.2f} t/s ({r['elapsed']:6.2f}s)")
        
        print(f"\n{'Average':30} {avg_tps:8.2f} t/s")
        print(f"{'Min':30} {min_tps:8.2f} t/s")
        print(f"{'Max':30} {max_tps:8.2f} t/s")
        
        print("\n✓ Benchmark complete!")
    else:
        print("\n✗ No successful benchmarks")
        sys.exit(1)


def main():
    print("\n" + "="*70)
    print("CODE REVIEW CRITIC - CPU BENCHMARK")
    print("="*70)
    
    # Check prerequisites
    if not check_ollama():
        print("\nTo install Ollama:")
        print("  Visit: https://ollama.ai")
        print("  Or: brew install ollama (macOS)")
        sys.exit(1)
    
    if not check_model_loaded():
        print("\nTo load the model, run:")
        print(f"  cd {Path(__file__).parent}")
        print(f"  ollama create code-review-critic -f Modelfile")
        sys.exit(1)
    
    # Run benchmark
    benchmark_cpu_inference()


if __name__ == "__main__":
    main()
