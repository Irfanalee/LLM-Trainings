"""Frame-level AUROC evaluation on UCSD Ped2.

Scores every test frame, compares against ground truth labels,
and computes the publishable AUROC number.

Expected test directory layout:
    data/ucsd_ped2/
    ├── Test/
    │   ├── Test001/   ← frames: 001.tif, 002.tif, ...
    │   ├── Test002/
    │   └── ...
    └── Test_gt/       ← ground truth (one of two formats):
        ├── Test001_gt/   ← binary .tif masks per frame
        └── Test001_gt.mat ← MATLAB array (n_frames,)

Usage:
    python eval/eval_roc.py \
        --checkpoint checkpoints/train_default_epoch027_val0.0317.pt \
        --test-dir data/ucsd_ped2 \
        --output outputs/eval/
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.augmentations import NormalizeVideo
from models.jepa_model import JEPAWorldModel
from training.utils import load_checkpoint


# ---------------------------------------------------------------------------
# Ground truth loading — supports both .tif mask dirs and .mat files
# ---------------------------------------------------------------------------

def _load_gt_from_masks(gt_dir: str) -> np.ndarray:
    """Load frame-level GT from a folder of binary .tif masks."""
    mask_files = sorted(
        f for f in os.listdir(gt_dir) if f.lower().endswith(".tif")
    )
    labels = []
    for fname in mask_files:
        mask = np.array(Image.open(os.path.join(gt_dir, fname)))
        labels.append(1 if mask.any() else 0)
    return np.array(labels, dtype=np.int32)


def _load_gt_from_mat(mat_path: str) -> np.ndarray:
    """Load frame-level GT from a .mat file."""
    import scipy.io
    mat = scipy.io.loadmat(mat_path)
    # Find the first non-metadata key with a numeric array
    for key, val in mat.items():
        if key.startswith("_"):
            continue
        arr = np.array(val).flatten()
        if arr.dtype.kind in ("i", "u", "f") and arr.size > 0:
            return (arr > 0).astype(np.int32)
    raise ValueError(f"Could not find GT array in {mat_path}")


def load_gt_labels(test_dir: str, video_name: str) -> np.ndarray:
    """Load ground truth labels for a test video. Returns (n_frames,) int array.

    Supports three GT formats:
      1. _gt/ subdirectory with binary .bmp/.tif masks (UCSD Ped2)
      2. _gt.mat file (MATLAB format)
      3. .npy file (ShanghaiTech test_frame_mask/)
    """
    gt_base = os.path.join(test_dir, "Test_gt")

    # Try 1: _gt subdirectory with binary mask images (UCSD Ped2)
    gt_dir = os.path.join(gt_base, f"{video_name}_gt")
    if os.path.isdir(gt_dir):
        return _load_gt_from_masks(gt_dir)

    # Try 2: _gt.mat file
    mat_path = os.path.join(gt_base, f"{video_name}_gt.mat")
    if os.path.isfile(mat_path):
        return _load_gt_from_mat(mat_path)

    # Try 3: mat file directly next to Test/ dir
    mat_path2 = os.path.join(test_dir, f"{video_name}_gt.mat")
    if os.path.isfile(mat_path2):
        return _load_gt_from_mat(mat_path2)

    # Try 4: _gt folder inside Test/ dir (UCSD Ped2 layout where GT lives next to frames)
    gt_dir2 = os.path.join(test_dir, "Test", f"{video_name}_gt")
    if os.path.isdir(gt_dir2):
        return _load_gt_from_masks(gt_dir2)

    # Try 5: ShanghaiTech .npy frame mask (test_frame_mask/{video_name}.npy)
    npy_path = os.path.join(test_dir, "test_frame_mask", f"{video_name}.npy")
    if os.path.isfile(npy_path):
        arr = np.load(npy_path).flatten()
        return (arr > 0).astype(np.int32)

    raise FileNotFoundError(
        f"No GT found for {video_name}. "
        f"Checked: {gt_dir}, {gt_dir2}, {mat_path}, {mat_path2}, {npy_path}"
    )


# ---------------------------------------------------------------------------
# Frame loading
# ---------------------------------------------------------------------------

def load_video_frames(video_dir: str, frame_size: int) -> torch.Tensor:
    """Load all .tif frames from a directory. Returns (T, C, H, W) float32 [0,1]."""
    frame_files = sorted(
        f for f in os.listdir(video_dir)
        if f.lower().endswith((".tif", ".tiff", ".png", ".jpg"))
    )
    if not frame_files:
        raise FileNotFoundError(f"No image frames found in {video_dir}")

    tensors = []
    for fname in frame_files:
        img = Image.open(os.path.join(video_dir, fname)).convert("RGB")
        img = img.resize((frame_size, frame_size), resample=Image.BILINEAR)
        tensors.append(TF.to_tensor(img))
    return torch.stack(tensors)  # (T, 3, H, W)


# ---------------------------------------------------------------------------
# Clip scoring → frame scores
# ---------------------------------------------------------------------------

def score_video_frames(
    model,
    frames: torch.Tensor,
    clip_frames: int,
    ctx_frames: int,
    device: torch.device,
    normalizer: NormalizeVideo,
) -> np.ndarray:
    """Score all frames in a video using stride-1 clips.

    Returns (n_frames,) float array of per-frame anomaly scores.
    Each frame score is the mean over all clips that contain it.
    """
    n_frames = frames.shape[0]
    frame_scores = np.zeros(n_frames, dtype=np.float64)
    frame_counts = np.zeros(n_frames, dtype=np.int32)

    frames_norm = normalizer(frames)  # normalize once for the whole video

    n_clips = max(0, n_frames - clip_frames + 1)
    for clip_start in range(n_clips):
        clip = frames_norm[clip_start: clip_start + clip_frames]  # (16, C, H, W)
        context = clip[:ctx_frames].unsqueeze(0).to(device)       # (1, 12, C, H, W)
        target = clip[ctx_frames:].unsqueeze(0).to(device)        # (1, 4, C, H, W)

        with torch.no_grad():
            output = model(context, target)

        score = output.loss.item()
        # All frames in this clip share this score
        for f in range(clip_start, clip_start + clip_frames):
            frame_scores[f] += score
            frame_counts[f] += 1

    # Avoid divide-by-zero for the last (clip_frames-1) frames if stride>1
    valid = frame_counts > 0
    frame_scores[valid] /= frame_counts[valid]
    # Frames not covered by any clip get the mean score
    if not valid.all():
        frame_scores[~valid] = frame_scores[valid].mean()

    return frame_scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="WorldGuard frame-level AUROC evaluation"
    )
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint")
    parser.add_argument("--test-dir", required=True,
                        help="Dataset root. UCSD Ped2: contains Test/ and Test_gt/. "
                             "ShanghaiTech: contains frames/ and test_frame_mask/")
    parser.add_argument("--output", default="outputs/eval",
                        help="Directory to save ROC curve PNG and results")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load model ---
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model = JEPAWorldModel(config).to(device)
    load_checkpoint(args.checkpoint, model)
    model.eval()

    clip_frames = config["data"]["clip_frames"]    # 16
    ctx_frames = config["data"]["context_frames"]  # 12
    frame_size = config["data"]["frame_size"]      # 224
    normalizer = NormalizeVideo()

    # --- Auto-detect dataset layout ---
    # UCSD Ped2: test_dir/Test/<video_name>/
    # ShanghaiTech: test_dir/frames/<video_name>/
    if os.path.isdir(os.path.join(args.test_dir, "Test")):
        test_videos_dir = os.path.join(args.test_dir, "Test")
    elif os.path.isdir(os.path.join(args.test_dir, "frames")):
        test_videos_dir = os.path.join(args.test_dir, "frames")
    else:
        raise FileNotFoundError(
            f"Could not find Test/ or frames/ subdirectory in {args.test_dir}"
        )

    video_names = sorted(
        d for d in os.listdir(test_videos_dir)
        if os.path.isdir(os.path.join(test_videos_dir, d))
        and not d.endswith("_gt")
    )

    if not video_names:
        raise FileNotFoundError(f"No test video folders found in {test_videos_dir}")

    print(f"Found {len(video_names)} test videos in {test_videos_dir}")

    all_scores = []
    all_labels = []

    for video_name in video_names:
        video_dir = os.path.join(test_videos_dir, video_name)
        print(f"  Scoring {video_name} ...", end=" ", flush=True)

        try:
            frames = load_video_frames(video_dir, frame_size)
        except FileNotFoundError as e:
            print(f"SKIP ({e})")
            continue

        frame_scores = score_video_frames(
            model, frames, clip_frames, ctx_frames, device, normalizer
        )

        try:
            gt_labels = load_gt_labels(args.test_dir, video_name)
        except FileNotFoundError as e:
            print(f"SKIP GT ({e})")
            continue

        # Align lengths (GT and frames may differ by 1–2 frames)
        min_len = min(len(frame_scores), len(gt_labels))
        all_scores.extend(frame_scores[:min_len].tolist())
        all_labels.extend(gt_labels[:min_len].tolist())
        print(f"{min_len} frames, {gt_labels[:min_len].sum()} anomaly frames")

    if not all_scores:
        raise RuntimeError("No videos were scored. Check --test-dir structure.")

    y_true = np.array(all_labels)
    y_score = np.array(all_scores)

    # --- AUROC ---
    auroc = roc_auc_score(y_true, y_score)

    # --- ROC curve ---
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    # Equal Error Rate (where FPR ≈ FNR = 1 - TPR)
    fnr = 1.0 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0

    # Best threshold via Youden J = TPR - FPR
    youden_idx = np.argmax(tpr - fpr)
    best_threshold = thresholds[youden_idx]

    print(f"\n{'='*40}")
    print(f"  Frame AUROC      : {auroc:.4f}")
    print(f"  Equal Error Rate : {eer:.4f}")
    print(f"  Best threshold   : {best_threshold:.6f}  (Youden J)")
    print(f"{'='*40}\n")

    # --- Plot ROC curve ---
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="steelblue", lw=2,
            label=f"WorldGuard (AUC = {auroc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
    ax.scatter(fpr[eer_idx], tpr[eer_idx], color="orange", zorder=5,
               label=f"EER = {eer:.3f}")
    ax.scatter(fpr[youden_idx], tpr[youden_idx], color="red", zorder=5,
               label=f"Best threshold = {best_threshold:.4f}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("WorldGuard — ROC Curve (UCSD Ped2, Frame-level)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    roc_path = os.path.join(args.output, "roc_curve.png")
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)
    print(f"ROC curve saved to {roc_path}")


if __name__ == "__main__":
    main()
