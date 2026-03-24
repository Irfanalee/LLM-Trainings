# Code Review Critic — 20-Min Presentation

---

## 1. The Problem (~2 min)

- Code review is time-consuming and inconsistent
- Reviewers miss bugs, focus on style instead of substance
- Small teams / solo devs have no reviewer at all
- **Goal:** An LLM that gives constructive, actionable code review — not just linting

---

## 2. What We Built (~2 min)

- A fine-tuned LLM that reviews Python code like a senior engineer
- Identifies **bugs**, **potential issues**, and **improvements**
- Can run entirely on **CPU** (no GPU needed in production)
- Can review real GitHub PRs automatically

**Example:**
```python
# Input
def get_user(user_id):
    user = db.query(User).filter(id=user_id).first()
    return user.name

# Model Output
return user.name if user else None
```

---

## 3. Architecture (~3 min)

| Component | Choice |
|-----------|--------|
| Base Model | Qwen2.5-Coder-7B-Instruct |
| Training Method | QLoRA (4-bit quantization + LoRA adapters) |
| Training Hardware | NVIDIA RTX A4000 16GB |
| Export Format | Merged 16-bit → GGUF Q4 (~4GB) |
| Inference Runtime | Ollama / llama.cpp |
| Production Requirements | CPU only, 16GB RAM |

- **QLoRA** = train a 7B model on a 16GB GPU by quantizing to 4-bit and adding small trainable adapters
- **GGUF** = compressed format for CPU inference

---

## 4. The Data Pipeline (~4 min)

### Step 1 — Real GitHub PR Comments
- Scraped **10,809 comments** from high-quality Python repos (Django, FastAPI, PyTorch, Airflow, etc.)
- Filtered down to **7,012 quality examples**

### Step 2 — Synthetic Data (Bug Detection)
- Generated **1,500 targeted bug-detection examples** using Claude Haiku
- Cost: ~**$1.50**
- Plugged gaps the real data didn't cover well

### Step 3 — Data Cleaning
- Removed **237 contaminated examples** ("author response" comments that looked like review but weren't)
- Final dataset: **8,275 training examples + 780 eval examples**

### Data Format (ChatML)
```
System: You are an expert code reviewer...
User: Review this Python code from `file.py`: ...
Assistant: {review comment}
```

---

## 5. Training (~3 min)

| Config | Value |
|--------|-------|
| Max sequence length | 1,536 tokens |
| LoRA rank | 64 |
| Learning rate | 2e-4 |
| Effective batch size | 16 (gradient accumulation) |
| Epochs | 2 |
| Runtime | ~1.5–2 hours |

### Results
| Metric | Value |
|--------|-------|
| Final train loss | 0.82 |
| Final eval loss | 0.845 |

- Eval loss decreased consistently → no overfitting
- `1.11 → 0.98 → 0.90 → 0.86 → 0.845`

---

## 6. Deployment (~2 min)

### GPU Path
```bash
python review_pr.py https://github.com/owner/repo/pull/123
```

### CPU Path (Ollama)
- Model exported to **GGUF Q4** format (~4-5 GB)
- Loaded into Ollama for local CPU inference
```bash
ollama run code-review-critic "Review: def get_user(id): ..."
python review_pr_ollama.py https://github.com/owner/repo/pull/123
```

---

## 7. Key Lessons Learned (~2 min)

1. **Data quality > quantity** — 2% bad examples (author responses) caused hallucinations
2. **Overfitting is fast** — 3+ epochs spiked eval loss; 2 is optimal
3. **Synthetic data works** — $1.50 of Haiku-generated examples meaningfully improved bug detection
4. **Always keep checkpoints** — merge process can fail; checkpoints saved the project
5. **Dropout = 0** — required for Unsloth's 16GB VRAM optimizations

---

## 8. What's Next (~1 min)

- [ ] Docker image with Ollama for easy deployment
- [ ] GitHub Action for automated PR reviews on every push
- [ ] Improve uncertain/conversational responses ("I'm not sure...")
- [ ] Expand beyond Python to other languages

---

## 9. Demo (~1 min)

```bash
# Run against a real PR
python review_pr_ollama.py https://github.com/HKUDS/nanobot/pull/109
```
