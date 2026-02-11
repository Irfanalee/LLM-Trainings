#!/usr/bin/env python3
"""
Quick CPU Benchmark - Measures actual inference on your CPU
Uses llama.cpp directly if available, or provides setup instructions
"""

import subprocess
import sys
import os
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent
GGUF_MODEL = SCRIPT_DIR / "output/gguf/code-review-critic-q4_k_m.gguf"


def check_llama_cpp():
    """Check if llama.cpp is available"""
    try:
        result = subprocess.run(
            ["which", "llama-cli"],
            capture_output=True,
            timeout=2
        )
        if result.returncode == 0:
            return result.stdout.decode().strip()
    except:
        pass
    
    # Check common installation paths
    for path in [
        Path.home() / "llama.cpp" / "llama-cli",
        Path.home() / "llama.cpp" / "build" / "bin" / "llama-cli",
        "/usr/local/bin/llama-cli",
        "/opt/llama.cpp/llama-cli",
    ]:
        if path.exists():
            return str(path)
    
    return None


def run_benchmark():
    """Run performance benchmark using llama.cpp"""
    
    if not GGUF_MODEL.exists():
        print(f"ERROR: GGUF model not found at {GGUF_MODEL}")
        print(f"Available files:")
        gguf_dir = GGUF_MODEL.parent
        if gguf_dir.exists():
            for f in gguf_dir.iterdir():
                print(f"  - {f.name} ({f.stat().st_size / 1e9:.1f} GB)")
        return False
    
    print("\n" + "="*70)
    print("CPU PERFORMANCE BENCHMARK")
    print("="*70)
    print(f"\nModel: {GGUF_MODEL.name}")
    print(f"Size: {GGUF_MODEL.stat().st_size / 1e9:.1f} GB")
    
    # Find llama-cli
    llama_cli = check_llama_cpp()
    
    if not llama_cli:
        print("\n✗ llama-cpp-python or llama.cpp not found")
        print("\nTo measure CPU performance, you need llama.cpp:")
        print("\n1. Clone llama.cpp:")
        print("   git clone https://github.com/ggerganov/llama.cpp")
        print("   cd llama.cpp")
        print("\n2. Build it:")
        print("   cmake -B build")
        print("   cmake --build build --config Release")
        print("\n3. Run benchmark:")
        print(f"   ./build/bin/llama-bench -m {GGUF_MODEL}")
        return False
    
    print(f"\n✓ Found llama-cli at: {llama_cli}")
    
    # Run the benchmark
    print("\nRunning benchmark (this may take a few minutes)...")
    print("-" * 70)
    
    try:
        # Run llama-bench for comprehensive benchmarking
        result = subprocess.run(
            [llama_cli, 
             "-m", str(GGUF_MODEL),
             "-n", "256",  # 256 tokens
             "-t", str(os.cpu_count()),  # Use all CPU cores
             "-e",  # Evaluate embeddings
             "--verbose-prompt"],
            capture_output=False,
            text=True,
            timeout=600
        )
        
        if result.returncode == 0:
            print("\n" + "="*70)
            print("✓ Benchmark complete!")
            return True
        else:
            print("\n✗ Benchmark failed")
            return False
            
    except subprocess.TimeoutExpired:
        print("\n✗ Benchmark timeout")
        return False
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


def show_quick_test():
    """Show how to do a quick manual test"""
    print("\n" + "="*70)
    print("QUICK MANUAL TEST")
    print("="*70)
    
    llama_cli = check_llama_cpp()
    if llama_cli:
        print(f"\nRun this for a quick performance test:")
        print(f"\n  {llama_cli} \\")
        print(f"    -m {GGUF_MODEL} \\")
        print(f"    -n 128 \\")
        print(f"    -t {os.cpu_count()} \\")
        print(f'    -p "Review this code: def foo(): pass"')
    else:
        print("\nAfter installing llama.cpp, run:")
        print(f"\n  ./llama-cli \\")
        print(f"    -m {GGUF_MODEL} \\")
        print(f"    -n 256 \\")
        print(f"    -t {os.cpu_count()}")


def main():
    print("\n" + "="*70)
    print("CODE REVIEW CRITIC - CPU BENCHMARK")
    print("="*70)
    
    success = run_benchmark()
    
    if not success:
        show_quick_test()
    
    print("\n" + "="*70)
    print("EXPECTED PERFORMANCE ON CPU")
    print("="*70)
    print("""
For a 7B Qwen model (Q4_K_M quantization) on typical CPUs:
  - Intel i5/i7 (8 cores): ~2-5 tokens/s
  - AMD Ryzen 7 (8 cores): ~2-5 tokens/s
  - M1/M2 Mac: ~5-10 tokens/s
  - High-end Threadripper: ~10-20 tokens/s

Your actual speed depends on:
  ✓ Number of CPU cores (more = faster)
  ✓ CPU frequency (higher = faster)
  ✓ RAM speed and availability
  ✓ Other processes running concurrently
""")


if __name__ == "__main__":
    main()
