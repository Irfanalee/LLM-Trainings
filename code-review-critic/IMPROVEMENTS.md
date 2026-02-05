# Code Review Critic - Improvements

## Current Issues

### 1. Uncertain/Conversational Responses
The model sometimes produces vague, uncertain responses instead of actionable code reviews.

**Examples:**
- "Not sure, I will try it out and see if it works"
- "I am not sure this is correct"
- "I'm not sure I follow - are you suggesting we add the error handling?"

**Likely cause:** Training data contains conversational patterns or clarification requests from reviewers.

### 2. Inconsistent Quality
- ~1/3 reviews are good (specific, actionable suggestions)
- ~2/3 reviews are weak (vague, uncertain, or conversational)

---

## Potential Fixes

### Quick Fixes (No Retraining)

| Fix | Where | Change |
|-----|-------|--------|
| Lower temperature | `review_pr.py:76` | `temperature=0.4` (from 0.7) |
| Stronger system prompt | `review_pr.py:60-62` | Add "Never say 'I'm not sure'. Always provide specific feedback." |
| Multiple generations | `review_pr.py` | Generate 3 reviews, pick best one |

### Data Cleaning (Requires Retraining)

1. **Find uncertain patterns** in training data:
   ```bash
   grep -i "not sure\|i think\|maybe\|i'm not\|i am not" data/processed/train_cleaned.jsonl
   ```

2. **Remove or rephrase** examples with uncertain language

3. **Add more high-quality synthetic examples** with confident, specific reviews

### Model/Training Changes

| Change | Expected Impact |
|--------|-----------------|
| Increase LoRA rank (64→128) | More capacity, better pattern learning |
| More epochs (2→3) with early stopping | Better convergence (watch for overfitting) |
| Larger base model (7B→14B) | Better reasoning (needs 32GB+ VRAM) |

---

## Testing Checklist

After any fix, test with:

```bash
# Quick test
python test_model.py

# Real PR test
python review_pr.py https://github.com/HKUDS/nanobot/pull/109
```

**Success criteria:**
- [ ] No "I'm not sure" or similar uncertain phrases
- [ ] All reviews contain specific, actionable feedback
- [ ] Reviews reference actual code from the input

---

## Priority

1. **High:** Data cleaning (find and remove uncertain examples)
2. **Medium:** Lower temperature + stronger prompt
3. **Low:** Increase model capacity or base model size
