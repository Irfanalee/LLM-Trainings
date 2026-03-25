"""Threshold calibration for WorldGuard.

Runs a trained checkpoint over normal-only val clips, collects per-clip anomaly
scores, and writes a threshold JSON for the given camera.

Usage:
    python training/calibrate.py \
        --checkpoint checkpoints/train_default_epoch050_val0.1191.pt \
        --val-dir data/val \
        --camera-id cam01 \
        --multiplier 2.5
"""

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.dataset import ClipDataset
from models.jepa_model import JEPAWorldModel
from training.utils import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="WorldGuard threshold calibration")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint")
    parser.add_argument("--val-dir", required=True, help="Directory of normal-only val clips")
    parser.add_argument("--camera-id", required=True, help="Camera identifier (e.g. cam01)")
    parser.add_argument("--multiplier", type=float, default=2.5,
                        help="Threshold = mean + multiplier * std (default: 2.5)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load checkpoint (read-only: only model weights are used) ---
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["config"]

    model = JEPAWorldModel(config).to(device)
    load_checkpoint(args.checkpoint, model)   # loads weights; no optimizer/scaler
    model.eval()

    # --- Val dataset — no labels, no augmentation ---
    val_ds = ClipDataset(args.val_dir, config, augment=False)
    val_loader = DataLoader(
        val_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )

    # --- Collect per-clip scores ---
    all_scores = []
    with torch.no_grad():
        for context, target in val_loader:
            context = context.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            errors = model.patch_errors(context, target)   # (B, N)
            clip_scores = errors.mean(dim=1)               # (B,)
            all_scores.extend(clip_scores.cpu().tolist())

    if not all_scores:
        raise RuntimeError(f"No clips found in {args.val_dir}")

    scores = torch.tensor(all_scores)
    mean_score = scores.mean().item()
    std_score = scores.std(correction=0).item()
    threshold = mean_score + args.multiplier * std_score

    # --- Print summary ---
    print(f"\nScore distribution over {len(all_scores)} clips:")
    print(f"  min      : {scores.min().item():.6f}")
    print(f"  max      : {scores.max().item():.6f}")
    print(f"  mean     : {mean_score:.6f}")
    print(f"  std      : {std_score:.6f}")
    print(f"  threshold: {threshold:.6f}  (mean + {args.multiplier} * std)")

    # --- Write threshold JSON ---
    out_dir = os.path.join("configs", "thresholds")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.camera_id}.json")

    payload = {
        "camera_id": args.camera_id,
        "threshold": threshold,
        "mean_score": mean_score,
        "std_score": std_score,
        "num_clips": len(all_scores),
        "multiplier": args.multiplier,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nThreshold written to {out_path}")


if __name__ == "__main__":
    main()
