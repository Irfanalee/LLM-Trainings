"""WorldGuard training loop.

Usage:
    python training/train.py --config configs/train_default.yaml
    python training/train.py --config configs/train_default.yaml --dummy   # smoke test
    python training/train.py --config configs/train_default.yaml --resume checkpoints/ckpt.pt
"""

import argparse
import os
import sys

import torch
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.jepa_model import JEPAWorldModel
from training.utils import WandbLogger, load_checkpoint, save_checkpoint, update_ema


# ---------------------------------------------------------------------------
# Dummy dataset for smoke testing (--dummy flag)
# ---------------------------------------------------------------------------

class DummyDataset(Dataset):
    """Returns random tensors. Used for --dummy smoke tests only."""

    def __init__(self, config: dict, size: int = 64) -> None:
        self.size = size
        self.ctx_frames = config["data"]["context_frames"]
        self.tgt_frames = config["data"]["target_frames"]
        self.frame_size = config["data"]["frame_size"]

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        ctx = torch.randn(self.ctx_frames, 3, self.frame_size, self.frame_size)
        tgt = torch.randn(self.tgt_frames, 3, self.frame_size, self.frame_size)
        return ctx, tgt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_target_enc_not_in_optimizer(model: JEPAWorldModel, optimizer) -> None:
    """Safety check: target_encoder params must never receive gradients via optimizer."""
    optimizer_param_ids = {
        id(p) for group in optimizer.param_groups for p in group["params"]
    }
    for p in model.target_encoder.parameters():
        assert id(p) not in optimizer_param_ids, (
            "target_encoder parameter found in optimizer! "
            "Target encoder must be EMA-only and never trained directly."
        )


def _build_loaders(config: dict, dummy: bool):
    if dummy:
        train_ds = DummyDataset(config, size=64)
        val_ds = DummyDataset(config, size=32)
    else:
        from data.dataset import ClipDataset
        train_ds = ClipDataset(config["data"]["train_dir"], config, augment=True)
        val_ds = ClipDataset(config["data"]["val_dir"], config, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"] if not dummy else 0,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"] if not dummy else 0,
        pin_memory=True,
    )
    return train_loader, val_loader


def _run_validation(model: JEPAWorldModel, val_loader: DataLoader, device: torch.device):
    """Return (mean_score, std_score) over val set. No labels used."""
    model.eval()
    scores = []
    with torch.no_grad():
        for context, target in val_loader:
            context = context.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            with autocast():
                output = model(context, target)
            # Per-clip loss detached from graph
            scores.append(output.loss.item())
    model.train()
    score_tensor = torch.tensor(scores)
    # correction=0 (population std) avoids NaN when val set yields only 1 batch
    return score_tensor.mean().item(), score_tensor.std(correction=0).item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="WorldGuard JEPA training loop")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--dummy", action="store_true", help="Use random data (smoke test)")
    args = parser.parse_args()

    # --- Config ---
    with open(args.config) as f:
        config = yaml.safe_load(f)

    config_name = os.path.splitext(os.path.basename(args.config))[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = config["training"]["epochs"]
    warmup_epochs = config["training"]["warmup_epochs"]
    grad_clip = config["training"]["grad_clip"]
    log_interval = config["logging"]["log_interval"]
    ckpt_dir = config["checkpoints"]["dir"]
    save_interval = config["checkpoints"]["save_interval"]

    # --- Model ---
    model = JEPAWorldModel(config).to(device)
    model.train()

    # --- Optimizer: context_encoder + predictor only ---
    trainable_params = (
        list(model.context_encoder.parameters())
        + list(model.predictor.parameters())
    )
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )

    # Critical safety check — target_encoder must never be optimized directly
    _assert_target_enc_not_in_optimizer(model, optimizer)

    # --- LR schedule: linear warmup → cosine decay ---
    warmup_sched = LinearLR(
        optimizer,
        start_factor=1.0 / max(warmup_epochs, 1),
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine_sched = CosineAnnealingLR(
        optimizer,
        T_max=max(epochs - warmup_epochs, 1),
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup_epochs],
    )

    # --- Mixed precision ---
    scaler = GradScaler()

    # --- Resume ---
    start_epoch = 0
    best_val_score = float("inf")
    if args.resume:
        start_epoch, best_val_score = load_checkpoint(
            args.resume, model, optimizer, scaler
        )
        start_epoch += 1  # resume from next epoch
        print(f"Resumed from {args.resume} at epoch {start_epoch}, best val {best_val_score:.4f}")

    # --- Data ---
    train_loader, val_loader = _build_loaders(config, dummy=args.dummy)

    # --- W&B ---
    logger = WandbLogger(config, config_name)

    # --- Training loop ---
    global_step = 0
    print(f"Starting training on {device}. Epochs: {epochs}, batch: {config['training']['batch_size']}")

    for epoch in range(start_epoch, epochs):
        for context, target in train_loader:
            context = context.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast():
                output = model(context, target)          # forward

            scaler.scale(output.loss).backward()         # backward

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)

            scaler.step(optimizer)                       # optimizer.step()
            scaler.update()

            update_ema(model)                            # EMA — always last

            global_step += 1
            if global_step % log_interval == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                logger.log_step(output.loss.item(), current_lr, global_step)
                print(
                    f"Epoch {epoch+1}/{epochs} | step {global_step} | "
                    f"loss {output.loss.item():.4f} | lr {current_lr:.2e}"
                )

        # Per-epoch LR step
        scheduler.step()

        # Validation
        val_mean, val_std = _run_validation(model, val_loader, device)
        logger.log_epoch(val_mean, val_std, epoch + 1)
        print(f"Epoch {epoch+1} val | mean {val_mean:.4f} | std {val_std:.4f}")

        # Save best checkpoint
        if val_mean < best_val_score:
            best_val_score = val_mean
            path = save_checkpoint(
                model, optimizer, scaler,
                epoch + 1, val_mean, config, config_name, ckpt_dir,
            )
            print(f"  New best checkpoint: {path}")

        # Periodic checkpoint
        elif (epoch + 1) % save_interval == 0:
            save_checkpoint(
                model, optimizer, scaler,
                epoch + 1, val_mean, config, config_name, ckpt_dir,
            )

    logger.finish()
    print("Training complete.")


if __name__ == "__main__":
    main()
