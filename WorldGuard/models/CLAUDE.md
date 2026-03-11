# CLAUDE.md — models/

## Critical Rules for This Directory

This is the core of WorldGuard. Be extremely careful here.

### The One Rule That Must Never Break
**The target encoder must NEVER receive gradients.**

```python
# CORRECT
with torch.no_grad():
    z_tgt = self.target_encoder(target_frames)

# WRONG — will train the target encoder and break EMA stability
z_tgt = self.target_encoder(target_frames)
```

If you break this, training will appear to work (loss decreases) but the model will collapse
and produce garbage anomaly scores. It is very hard to detect after the fact.

### EMA Update Order
EMA update MUST happen AFTER `optimizer.step()`, not before.

```python
# Correct order in training loop:
loss.backward()
optimizer.step()
model.update_target_encoder()  # ← ALWAYS last
```

### Files in This Directory

| File | Purpose | Touch carefully? |
|---|---|---|
| `encoder.py` | ViT-S/16 + adapter. Shared by context + target encoder | YES — changes affect both encoders |
| `predictor.py` | 4-layer Transformer predictor | Moderate — architecture changes need VRAM re-check |
| `jepa_model.py` | Combines everything. Contains EMA logic | YES — core training stability lives here |

### When Adding Architecture Changes
1. Check VRAM budget first (see `docs/architecture.md`)
2. Run forward pass with dummy data before touching real data
3. Document the change in `docs/team-decisions.md` as a new ADR
