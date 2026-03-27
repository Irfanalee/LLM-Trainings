"""Score a raw video file for anomalies using a trained WorldGuard checkpoint.

For each 16-frame clip extracted from the video:
  1. Encode context frames, predict target latent, compare to actual target
  2. Compute clip anomaly score (mean patch L2 error)
  3. Compare against per-camera threshold from configs/thresholds/{camera_id}.json
  4. Save anomalous clips and a heatmap PNG to --output-dir

Optional Stage 2 feedback classifier (--feedback-classifier):
  5. For Stage 1 flagged clips, run z_pred through FeedbackClassifier
  6. Final alert only if Stage 2 probability > 0.5

Usage:
    # Stage 1 only:
    python inference/score_video.py \
        --video /path/to/video.mp4 \
        --checkpoint checkpoints/train_default_epoch050_val0.0191.pt \
        --camera-id cam01 \
        --output-dir outputs/

    # Stage 1 + Stage 2 feedback classifier:
    python inference/score_video.py \
        --video /path/to/video.mp4 \
        --checkpoint checkpoints/train_default_epoch050_val0.0191.pt \
        --camera-id cam01 \
        --feedback-classifier checkpoints/feedback_cam01.pt \
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
import torchvision.transforms.functional as TF
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.augmentations import NormalizeVideo
from inference.heatmap import generate_heatmap, overlay_heatmap
from models.feedback_classifier import FeedbackClassifier
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


def load_feedback_classifier(path: str, device: torch.device) -> FeedbackClassifier:
    """Load a trained FeedbackClassifier checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = FeedbackClassifier().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    val_acc = ckpt.get("val_acc", 0.0)
    n_labels = ckpt.get("n_labels", 0)
    print(f"Stage 2 classifier loaded | val_acc={val_acc:.3f} | trained on {n_labels} labels")
    return model


def score_clip(
    model,
    context: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
    device: torch.device,
) -> tuple[float, bool, np.ndarray, torch.Tensor]:
    """Score a single clip.

    Args:
        model:     JEPAWorldModel in eval mode.
        context:   (T_ctx, C, H, W) normalized context frames.
        target:    (T_tgt, C, H, W) normalized target frames.
        threshold: Per-camera anomaly threshold.
        device:    Compute device.

    Returns:
        (anomaly_score, is_anomaly, heatmap_224x224, z_pred)
    """
    ctx = context.unsqueeze(0).to(device)
    tgt = target.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(ctx, tgt)
        heatmap = generate_heatmap(model, ctx, tgt)

    score = output.loss.item()
    is_anomaly = score > threshold
    return score, is_anomaly, heatmap, output.z_pred


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
    parser.add_argument("--feedback-classifier", default=None,
                        help="Optional Stage 2 classifier checkpoint path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load JEPA model (Stage 1) ---
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model = JEPAWorldModel(config).to(device)
    load_checkpoint(args.checkpoint, model)
    model.eval()

    ctx_frames = config["data"]["context_frames"]   # 12
    clip_frames = config["data"]["clip_frames"]      # 16
    frame_size = config["data"]["frame_size"]        # 224

    normalizer = NormalizeVideo()

    # --- Load Stage 2 classifier (optional) ---
    feedback_model = None
    if args.feedback_classifier:
        feedback_model = load_feedback_classifier(args.feedback_classifier, device)

    # --- Load threshold ---
    threshold = load_threshold(args.camera_id)
    print(f"Camera: {args.camera_id}  |  threshold: {threshold:.6f}")

    # --- Prepare output dirs ---
    clips_dir = os.path.join(args.output_dir, "anomaly_clips")
    heatmaps_dir = os.path.join(args.output_dir, "heatmaps")
    embeddings_dir = os.path.join(args.output_dir, "embeddings")
    os.makedirs(clips_dir, exist_ok=True)
    os.makedirs(heatmaps_dir, exist_ok=True)
    os.makedirs(embeddings_dir, exist_ok=True)

    csv_path = os.path.join(args.output_dir, f"{args.camera_id}_scores.csv")
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["clip_idx", "start_frame", "score", "stage1_anomaly",
                     "stage2_prob", "final_anomaly"])

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

            clip_tensor = _frames_to_tensor(frame_buffer, frame_size)
            clip_tensor = normalizer(clip_tensor)

            context = clip_tensor[:ctx_frames]
            target = clip_tensor[ctx_frames:]

            score, stage1_anomaly, heatmap, z_pred = score_clip(
                model, context, target, threshold, device
            )

            # Stage 2: run feedback classifier if loaded and Stage 1 flagged
            stage2_prob = None
            final_anomaly = stage1_anomaly
            if stage1_anomaly and feedback_model is not None:
                with torch.no_grad():
                    stage2_prob = feedback_model(z_pred).item()
                final_anomaly = stage2_prob > 0.5

            writer.writerow([
                clip_idx, start_frame, f"{score:.6f}",
                int(stage1_anomaly),
                f"{stage2_prob:.4f}" if stage2_prob is not None else "",
                int(final_anomaly),
            ])

            if stage1_anomaly:
                # Always save embedding for human review / future training
                emb_path = os.path.join(
                    embeddings_dir, f"{args.camera_id}_clip{clip_idx:06d}_emb.pt"
                )
                torch.save(z_pred.squeeze(0).cpu(), emb_path)  # (196, 384)

                # Save heatmap and clip only for final alerts
                if final_anomaly:
                    anomaly_count += 1
                    mid_frame = np.array(
                        frame_buffer[ctx_frames // 2].resize((frame_size, frame_size))
                    )
                    overlay = overlay_heatmap(mid_frame, heatmap)
                    heatmap_path = os.path.join(
                        heatmaps_dir, f"{args.camera_id}_clip{clip_idx:06d}.png"
                    )
                    Image.fromarray(overlay).save(heatmap_path)

                    clip_path = os.path.join(
                        clips_dir, f"{args.camera_id}_clip{clip_idx:06d}_score{score:.4f}.pt"
                    )
                    torch.save(clip_tensor, clip_path)

                    stage2_str = f" | stage2={stage2_prob:.3f}" if stage2_prob is not None else ""
                    print(
                        f"  ANOMALY clip {clip_idx:06d} | "
                        f"frame {start_frame} | score {score:.4f}{stage2_str}"
                    )
                else:
                    print(
                        f"  filtered clip {clip_idx:06d} | "
                        f"frame {start_frame} | score {score:.4f} | "
                        f"stage2={stage2_prob:.3f} → false positive"
                    )

            # Slide window
            frame_buffer = frame_buffer[args.stride:]
            clip_idx += 1

            if clip_idx % 100 == 0:
                print(f"  Processed {clip_idx} clips, {anomaly_count} anomalies so far...")

    container.close()
    csv_file.close()

    print(f"\nDone. {clip_idx} clips scored, {anomaly_count} final anomalies.")
    print(f"Scores CSV   : {csv_path}")
    print(f"Heatmaps     : {heatmaps_dir}/")
    print(f"Anomaly clips: {clips_dir}/")
    print(f"Embeddings   : {embeddings_dir}/  ← label these with review_anomalies.py")


if __name__ == "__main__":
    main()
