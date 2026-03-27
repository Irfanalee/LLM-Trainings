"""Train the Stage 2 FeedbackClassifier on human-reviewed labels.

Loads labeled z_pred embeddings, trains a lightweight MLP binary classifier,
and saves the checkpoint for use in inference/score_video.py.

The JEPA model is never loaded or modified here — Stage 2 trains only on
the pre-saved z_pred embeddings.

Usage:
    python training/train_feedback.py \
        --labels-file data/feedback/labels.json \
        --embeddings-dir outputs/embeddings \
        --camera-id ucsd \
        --output checkpoints/feedback_ucsd.pt

Minimum recommended: ~20 labels per class (40 total).
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.feedback_dataset import FeedbackDataset
from models.feedback_classifier import FeedbackClassifier


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Dataset ---
    dataset = FeedbackDataset(args.labels_file, args.embeddings_dir, args.camera_id)
    n_total = len(dataset)
    n_val = max(1, int(n_total * 0.2))
    n_train = n_total - n_val

    if n_train < 2:
        print(f"Only {n_total} labelled clips — need at least 10. Label more clips first.")
        return

    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_set, batch_size=min(32, n_train), shuffle=True)
    val_loader = DataLoader(val_set, batch_size=min(32, n_val))

    # Count class balance
    all_labels = [dataset[i][1].item() for i in range(n_total)]
    n_pos = sum(all_labels)
    n_neg = n_total - n_pos
    print(f"Dataset: {n_total} clips | {int(n_pos)} true anomaly | {int(n_neg)} false positive")

    # Class-weighted loss to handle imbalance
    pos_weight = torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32).to(device)
    criterion = nn.BCELoss()

    # --- Model ---
    model = FeedbackClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    print(f"Training FeedbackClassifier for {args.epochs} epochs on {device}...\n")

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for z_pred, labels in train_loader:
            z_pred, labels = z_pred.to(device), labels.to(device)
            optimizer.zero_grad()
            probs = model(z_pred)
            loss = criterion(probs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Val
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for z_pred, labels in val_loader:
                z_pred, labels = z_pred.to(device), labels.to(device)
                probs = model(z_pred)
                preds = (probs > 0.5).float()
                correct += (preds == labels).sum().item()
                total += len(labels)

        val_acc = correct / total if total > 0 else 0.0
        avg_loss = train_loss / len(train_loader)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{args.epochs} | loss {avg_loss:.4f} | val_acc {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "camera_id": args.camera_id,
                "val_acc": val_acc,
                "n_labels": n_total,
            }, args.output)

    print(f"\nBest val accuracy: {best_val_acc:.3f}")
    print(f"Checkpoint saved to {args.output}")
    print(f"\nTo use in inference:")
    print(f"  python inference/score_video.py --video <path> --camera-id {args.camera_id} \\")
    print(f"    --feedback-classifier {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Stage 2 FeedbackClassifier"
    )
    parser.add_argument("--labels-file", required=True,
                        help="Path to labels JSON file")
    parser.add_argument("--embeddings-dir", required=True,
                        help="Directory containing *_emb.pt files")
    parser.add_argument("--camera-id", required=True,
                        help="Camera ID to train classifier for")
    parser.add_argument("--output", required=True,
                        help="Path to save classifier checkpoint")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs (default: 50)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
