"""t-SNE visualization of normal vs anomaly latent embeddings.

Collects z_pred embeddings from normal and anomaly clips, runs t-SNE,
and produces a clean scatter plot suitable for publication / LinkedIn.

Usage:
    python eval/visualize_errors.py \
        --checkpoint checkpoints/train_default_epoch027_val0.0317.pt \
        --normal-dir data/val \
        --anomaly-dir data/ucsd_ped2/Test \
        --anomaly-gt data/ucsd_ped2/Test_gt \
        --output outputs/eval/tsne.png \
        --n-samples 200
"""

import argparse
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.augmentations import NormalizeVideo
from data.dataset import ClipDataset
from eval.eval_roc import load_gt_labels, load_video_frames
from models.jepa_model import JEPAWorldModel
from training.utils import load_checkpoint


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_embedding(model, context: torch.Tensor, target: torch.Tensor, device) -> np.ndarray:
    """Extract a single (D,) embedding for one clip via mean-pooled z_pred."""
    ctx = context.unsqueeze(0).to(device)
    tgt = target.unsqueeze(0).to(device)
    output = model(ctx, tgt)
    # z_pred: (1, N, D) → mean over patches → (D,)
    embedding = output.z_pred.squeeze(0).mean(dim=0).cpu().float().numpy()
    return embedding


def collect_normal_embeddings(
    model,
    normal_dir: str,
    config: dict,
    device,
    n_samples: int,
) -> np.ndarray:
    """Collect embeddings from normal val clips. Returns (N, D)."""
    dataset = ClipDataset(normal_dir, config, augment=False)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    indices = indices[:n_samples]

    embeddings = []
    for idx in indices:
        context, target = dataset[idx]
        emb = extract_embedding(model, context, target, device)
        embeddings.append(emb)

    return np.stack(embeddings)


def collect_anomaly_embeddings(
    model,
    anomaly_dir: str,
    anomaly_gt_dir: str,
    config: dict,
    device,
    n_samples: int,
) -> np.ndarray:
    """Collect embeddings from anomaly clips (clips whose middle frame is anomalous)."""
    clip_frames = config["data"]["clip_frames"]
    ctx_frames = config["data"]["context_frames"]
    frame_size = config["data"]["frame_size"]
    normalizer = NormalizeVideo()

    video_names = sorted(
        d for d in os.listdir(anomaly_dir)
        if os.path.isdir(os.path.join(anomaly_dir, d)) and not d.endswith("_gt")
    )

    embeddings = []
    for video_name in video_names:
        if len(embeddings) >= n_samples:
            break

        video_dir = os.path.join(anomaly_dir, video_name)
        try:
            frames = load_video_frames(video_dir, frame_size)
            gt_labels = load_gt_labels(
                os.path.dirname(anomaly_dir), video_name
            )
        except (FileNotFoundError, ValueError):
            continue

        frames_norm = normalizer(frames)
        n_frames = frames.shape[0]
        mid_offset = clip_frames // 2  # middle frame of the clip

        for clip_start in range(max(0, n_frames - clip_frames + 1)):
            if len(embeddings) >= n_samples:
                break
            mid_frame = clip_start + mid_offset
            if mid_frame >= len(gt_labels):
                continue
            if gt_labels[mid_frame] != 1:
                continue  # skip non-anomaly clips

            clip = frames_norm[clip_start: clip_start + clip_frames]
            context = clip[:ctx_frames]
            target = clip[ctx_frames:]
            emb = extract_embedding(model, context, target, device)
            embeddings.append(emb)

    if not embeddings:
        raise RuntimeError(
            "No anomaly clips found. Check --anomaly-dir and --anomaly-gt paths."
        )
    return np.stack(embeddings[:n_samples])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="t-SNE visualization of WorldGuard latent space"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--normal-dir", required=True,
                        help="Directory of normal .pt clips (e.g. data/val)")
    parser.add_argument("--anomaly-dir", required=True,
                        help="UCSD Ped2 Test/ directory with frame subfolders")
    parser.add_argument("--anomaly-gt", required=True,
                        help="UCSD Ped2 Test_gt/ directory with GT labels")
    parser.add_argument("--output", default="outputs/eval/tsne.png")
    parser.add_argument("--n-samples", type=int, default=200,
                        help="Max clips per class for t-SNE (default: 200)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load model ---
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model = JEPAWorldModel(config).to(device)
    load_checkpoint(args.checkpoint, model)
    model.eval()

    print(f"Collecting normal embeddings (up to {args.n_samples}) ...")
    normal_emb = collect_normal_embeddings(
        model, args.normal_dir, config, device, args.n_samples
    )

    print(f"Collecting anomaly embeddings (up to {args.n_samples}) ...")
    anomaly_emb = collect_anomaly_embeddings(
        model, args.anomaly_dir, args.anomaly_gt, config, device, args.n_samples
    )

    print(f"Normal: {len(normal_emb)} clips | Anomaly: {len(anomaly_emb)} clips")

    # --- t-SNE ---
    all_emb = np.concatenate([normal_emb, anomaly_emb], axis=0)
    labels = np.array([0] * len(normal_emb) + [1] * len(anomaly_emb))

    print("Running t-SNE ...")
    tsne = TSNE(
        n_components=2,
        perplexity=min(30, len(all_emb) // 4),
        n_iter=1000,
        random_state=42,
    )
    coords = tsne.fit_transform(all_emb)  # (N, 2)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(9, 7))

    normal_mask = labels == 0
    anomaly_mask = labels == 1

    ax.scatter(
        coords[normal_mask, 0], coords[normal_mask, 1],
        c="steelblue", alpha=0.6, s=18, label=f"Normal ({normal_mask.sum()})",
    )
    ax.scatter(
        coords[anomaly_mask, 0], coords[anomaly_mask, 1],
        c="crimson", alpha=0.7, s=18, label=f"Anomaly ({anomaly_mask.sum()})",
    )

    ax.set_title(
        "WorldGuard — JEPA Latent Space (t-SNE)\n"
        "Unsupervised CCTV Anomaly Detection",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(args.output, dpi=150)
    plt.close(fig)

    print(f"t-SNE plot saved to {args.output}")


if __name__ == "__main__":
    main()
