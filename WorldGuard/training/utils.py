"""Training utilities for WorldGuard.

- update_ema: EMA update (must be called after optimizer.step())
- save_checkpoint / load_checkpoint
- WandbLogger: logs the 5 required metrics
"""

import os
import torch
import wandb


def update_ema(model) -> None:
    """Update target encoder via EMA. MUST be called after optimizer.step()."""
    model.update_target_encoder()


def save_checkpoint(
    model,
    optimizer,
    scaler,
    epoch: int,
    val_score: float,
    config: dict,
    config_name: str,
    ckpt_dir: str,
) -> str:
    """Save training checkpoint. Returns the saved file path."""
    os.makedirs(ckpt_dir, exist_ok=True)
    filename = f"{config_name}_epoch{epoch:03d}_val{val_score:.4f}.pt"
    path = os.path.join(ckpt_dir, filename)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "val_mean_score": val_score,
            "config": config,
        },
        path,
    )
    return path


def load_checkpoint(path: str, model, optimizer=None, scaler=None):
    """Load checkpoint. Returns (epoch, val_mean_score) for resuming."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt.get("epoch", 0), ckpt.get("val_mean_score", float("inf"))


class WandbLogger:
    """Logs the 5 required WorldGuard metrics to W&B."""

    def __init__(self, config: dict, config_name: str) -> None:
        wandb.init(
            project=config["logging"]["wandb_project"],
            entity=config["logging"].get("wandb_entity") or None,
            name=config_name,
            config=config,
        )

    def log_step(self, loss: float, lr: float, step: int) -> None:
        """Log per-step metrics: train/loss, train/lr, gpu/memory_mb."""
        wandb.log(
            {
                "train/loss": loss,
                "train/lr": lr,
                "gpu/memory_mb": torch.cuda.memory_allocated() / 1e6,
            },
            step=step,
        )

    def log_epoch(self, mean_score: float, std_score: float, epoch: int) -> None:
        """Log per-epoch validation metrics: val/mean_score, val/std_score."""
        wandb.log(
            {
                "val/mean_score": mean_score,
                "val/std_score": std_score,
            },
            step=epoch,
        )

    def finish(self) -> None:
        wandb.finish()
