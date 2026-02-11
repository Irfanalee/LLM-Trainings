# CPU Benchmark Scripts for Code Review Critic

Three scripts are available to measure CPU token/s output:

## 1. **benchmark_quick.py** (Recommended - Easiest to Use)
```bash
python3 benchmark_quick.py
```
- Automatically detects llama.cpp installation
- Shows expected performance for different CPUs
- Provides setup instructions if needed

## 2. **benchmark_cpu.py** (Using llama-cpp-python)
```bash
# First install:
pip install llama-cpp-python

# Then run:
python3 benchmark_cpu.py
```
- Direct Python binding to llama.cpp
- Detailed metrics (prompt processing vs token generation)
- Requires llama-cpp-python library

## 3. **benchmark_cpu_ollama.py** (Using Ollama Docker)
```bash
# First setup:
ollama create code-review-critic -f Modelfile

# Then run:
python3 benchmark_cpu_ollama.py
```
- Uses Ollama for easy deployment
- Best for production benchmarking
- Requires Ollama to be installed and running

---

## Quick Start (No Installation)

If you already have **llama.cpp** built, run directly:

```bash
# From llama.cpp directory
./build/bin/llama-cli \
  -m /path/to/code-review-critic-q4_k_m.gguf \
  -n 256 \
  -t $(nproc)
```

---

## Expected CPU Performance

For the **7B Qwen model (Q4_K_M quantization)**:

| CPU | Threads | Tokens/s |
|-----|---------|----------|
| Intel i5/i7 | 8 | 2-5 |
| AMD Ryzen 7 | 8 | 2-5 |
| Apple M1/M2 | 8 | 5-10 |
| Threadripper | 16+ | 10-20 |

**Note:** Actual performance depends on RAM speed, system load, and CPU frequency.

---

## What the Scripts Measure

### Prompt Processing Speed
- How fast the model processes input tokens
- Typically slower than generation (more computation needed)

### Token Generation Speed
- How fast the model generates output tokens
- The primary metric for inference performance

### Overall Throughput
- Average of both prompt and generation speed
- Real-world performance metric

---

## Interpreting Results

Look for lines like:
```
prompt eval time = 655.63 ms / 10 tokens (65.56 ms per token, 15.25 tokens per second)
       eval time = 2180.97 ms / 27 runs (80.78 ms per token, 12.38 tokens per second)
```

This shows:
- **Prompt speed:** 15.25 t/s (slower - processing input)
- **Generation speed:** 12.38 t/s (typical output rate)

---

## Installation Options

### Option A: llama.cpp (Recommended)
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build
cmake --build build --config Release
```

### Option B: llama-cpp-python
```bash
pip install llama-cpp-python
```

### Option C: Ollama
```bash
# macOS
brew install ollama

# Linux / Docker
docker run -d -v /path/to/models:/root/.ollama -p 11434:11434 ollama/ollama
```

---

## Optimization Tips

To improve CPU inference speed:

1. **Use all CPU cores:**
   ```bash
   -t $(nproc)  # Use all available cores
   ```

2. **Pin threads to cores (Linux):**
   ```bash
   numactl -C 0-7 ./llama-cli -m model.gguf
   ```

3. **Use faster quantization** (trades quality for speed):
   - Q5_K_M: Slower but better quality
   - Q4_K_M: Good balance (current)
   - Q4_K_S: Faster, less memory
   - Q3_K_M: Very fast, lower quality

4. **Reduce batch size** for lower latency:
   ```bash
   -n 128  # Smaller responses = faster
   ```

---

## Monitor System Performance

While benchmarking, monitor your system:

```bash
# CPU usage
top -bn1 | head -15

# Memory usage
free -h

# CPU frequency (Linux)
watch -n 1 'grep MHz /proc/cpuinfo | head -1'
```

---

## Next Steps

1. **Run one of the benchmark scripts**
2. **Note your tokens/s output**
3. **Share the results** if you want optimization advice
4. **Consider quantization changes** if too slow (see quantize_cpu.py)
