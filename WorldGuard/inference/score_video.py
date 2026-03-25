"""Score a raw video file for anomalies using a trained WorldGuard checkpoint.

For each 16-frame clip extracted from the video:
  1. Encode context frames, predict target latent, compare to actual target
  2. Compute clip anomaly score (mean patch L2 error)
  3. Compare against per-camera threshold from configs/thresholds/{camera_id}.json
  4. Save anomalous clips and a heatmap PNG to --output-dir

Usage:
    python inference/score_video.py \
        --video /path/to/video.mp4 \
        --checkpoint checkpoints/train_default_epoch027_val0.0317.pt \
        --camera-id cam01 \
        --output-dir outputs/
"""

import argparse
import csv
import json
import os
import sys

import av
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.augmentations import NormalizeVideo
from inference.heatmap import generate_heatmap, overlay_heatmap
from models.jepa_model import JEPAWorldModel
from training.utils import load_checkpoint


def load_threshold(camera_id: str) -> float:
    """Load per-camera threshold from configs/thresholds/{camera_id}.json."""
    path = os.path.join("configs", "thresholds", f"{camera_id}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No threshold file found at {path}. "
            f"Run training/calibrate.py --camera-id {camera_id} first."
        )
    with open(path) as f:
        data = json.load(f)
    return data["threshold"]


def score_clip(
    model,
    context: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
    device: torch.device,
) -> tuple[float, bool, np.ndarray]:
    """Score a single clip.

    Args:
        model:     JEPAWorldModel in eval mode.
        context:   (T_ctx, C, H, W) normalized context frames.
        target:    (T_tgt, C, H, W) normalized target frames.
        threshold: Per-camera anomaly threshold.
        device:    Compute device.

    Returns:
        (anomaly_score, is_anomaly, heatmap_224x224)
    """
    ctx = context.unsqueeze(0).to(device)
    tgt = target.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(ctx, tgt)
        heatmap = generate_heatmap(model, ctx, tgt)

    score = output.loss.item()
    is_anomaly = score > threshold
    return score, is_anomaly, heatmap


def _frames_to_tensor(frames: list, frame_size: int) -> torch.Tensor:
    """Convert list of PIL Images to a float32 tensor (T, C, H, W) in [0, 1]."""
    tensors = []
    for img in frames:
        img = img.resize((frame_size, frame_size), resample=Image.BILINEAR)
        tensors.append(TF.to_tensor(img))
    return torch.stack(tensors)  # (T, 3, H, W)


def main():
    parser = argparse.ArgumentParser(description="WorldGuard video anomaly scorer")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint")
    parser.add_argument("--camera-id", required=True, help="Camera ID (e.g. cam01)")
    parser.add_argument("--stride", type=int, default=2,
                        help="Frame stride for sliding window (default: 2)")
    parser.add_argument("--output-dir", default="outputs",
                        help="Directory to save anomaly clips and heatmaps")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load model ---
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model = JEPAWorldModel(config).to(device)
    load_checkpoint(args.checkpoint, model)
    model.eval()

    ctx_frames = config["data"]["context_frames"]   # 12
    tgt_frames = config["data"]["target_frames"]     # 4
    clip_frames = config["data"]["clip_frames"]      # 16
    frame_size = config["data"]["frame_size"]        # 224

    normalizer = NormalizeVideo()

    # --- Load threshold ---
    threshold = load_threshold(args.camera_id)
    print(f"Camera: {args.camera_id}  |  threshold: {threshold:.6f}")

    # --- Prepare output dirs ---
    clips_dir = os.path.join(args.output_dir, "anomaly_clips")
    heatmaps_dir = os.path.join(args.output_dir, "heatmaps")
    os.makedirs(clips_dir, exist_ok=True)
    os.makedirs(heatmaps_dir, exist_ok=True)

    csv_path = os.path.join(args.output_dir, f"{args.camera_id}_scores.csv")
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["clip_idx", "start_frame", "score", "is_anomaly"])

    # --- Extract and score clips via PyAV ---
    container = av.open(args.video)
    video_stream = container.streams.video[0]

    frame_buffer: list[Image.Image] = []
    frame_idx = 0
    clip_idx = 0
    anomaly_count = 0

    print(f"Scoring {args.video} ...")

    for frame in container.decode(video_stream):
        img: Image.Image = frame.to_image().convert("RGB")
        frame_buffer.append(img)
        frame_idx += 1

        if len(frame_buffer) == clip_frames:
            start_frame = frame_idx - clip_frames

            # Build tensor (T, C, H, W) in [0, 1]
            clip_tensor = _frames_to_tensor(frame_buffer, frame_size)
            clip_tensor = normalizer(clip_tensor)

            context = clip_tensor[:ctx_frames]   # (12, 3, 224, 224)
            target = clip_tensor[ctx_frames:]    # (4,  3, 224, 224)

            score, is_anomaly, heatmap = score_clip(
                model, context, target, threshold, device
            )

            writer.writerow([clip_idx, start_frame, f"{score:.6f}", int(is_anomaly)])

            if is_anomaly:
                anomaly_count += 1
                # Save heatmap overlaid on middle context frame
                mid_frame = np.array(
                    frame_buffer[ctx_frames // 2].resize((frame_size, frame_size))
                )
                overlay = overlay_heatmap(mid_frame, heatmap)
                heatmap_path = os.path.join(
                    heatmaps_dir, f"{args.camera_id}_clip{clip_idx:06d}.png"
                )
                Image.fromarray(overlay).save(heatmap_path)

                # Save raw clip tensor
                clip_path = os.path.join(
                    clips_dir, f"{args.camera_id}_clip{clip_idx:06d}_score{score:.4f}.pt"
                )
                torch.save(clip_tensor, clip_path)

                print(
                    f"  ANOMALY clip {clip_idx:06d} | "
                    f"frame {start_frame} | score {score:.4f} > {threshold:.4f}"
                )

            # Slide window
            frame_buffer = frame_buffer[args.stride:]
            clip_idx += 1

            if clip_idx % 100 == 0:
                print(f"  Processed {clip_idx} clips, {anomaly_count} anomalies so far...")

    container.close()
    csv_file.close()

    print(f"\nDone. {clip_idx} clips scored, {anomaly_count} anomalies detected.")
    print(f"Scores CSV  : {csv_path}")
    print(f"Heatmaps    : {heatmaps_dir}/")
    print(f"Anomaly clips: {clips_dir}/")


if __name__ == "__main__":
    main()
