"""Human review CLI for labelling flagged anomaly clips.

Renders each flagged clip as an animated GIF with heatmap overlay and opens it
in the system image viewer so you can see the motion before labelling.

Usage:
    python training/review_anomalies.py \
        --embeddings-dir outputs/embeddings \
        --clips-dir outputs/anomaly_clips \
        --heatmaps-dir outputs/heatmaps \
        --labels-file data/feedback/labels.json \
        --camera-id ucsd

Controls:
    t  → true anomaly  (label=1)
    f  → false positive (label=0)
    s  → skip (label not saved)
    q  → quit and save progress
"""

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image


def load_existing_labels(labels_file: str) -> set[str]:
    labelled = set()
    if not os.path.isfile(labels_file):
        return labelled
    with open(labels_file) as f:
        for line in f:
            line = line.strip()
            if line:
                labelled.add(json.loads(line)["clip"])
    return labelled


def clip_number(emb_file: str) -> str:
    """'ucsd_clip000042_emb.pt' → '000042'"""
    return emb_file.replace("_emb.pt", "").split("_clip")[-1]


def find_clip_tensor(clips_dir: str, camera_id: str, number: str) -> str | None:
    if not os.path.isdir(clips_dir):
        return None
    for fname in os.listdir(clips_dir):
        if fname.startswith(f"{camera_id}_clip{number}") and fname.endswith(".pt"):
            return os.path.join(clips_dir, fname)
    return None


def render_gif(clip_path: str, heatmap_path: str | None, out_path: str, fps: int = 8) -> None:
    """Render a clip tensor as an animated GIF with optional heatmap blend."""
    clip = torch.load(clip_path, map_location="cpu", weights_only=True)  # (T, 3, H, W)
    T, C, H, W = clip.shape

    hmap = None
    if heatmap_path and os.path.isfile(heatmap_path):
        hmap = np.array(Image.open(heatmap_path).convert("RGB").resize((W, H)))

    pil_frames = []
    for t in range(T):
        frame = (clip[t].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        if hmap is not None:
            frame = (frame * 0.6 + hmap * 0.4).clip(0, 255).astype(np.uint8)
        pil_frames.append(Image.fromarray(frame))

    duration_ms = int(1000 / fps)
    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )




def main() -> None:
    parser = argparse.ArgumentParser(
        description="Review anomaly clips as animated GIF and label for Stage 2 training"
    )
    parser.add_argument("--embeddings-dir", required=True)
    parser.add_argument("--clips-dir", default=None)
    parser.add_argument("--heatmaps-dir", default=None)
    parser.add_argument("--labels-file", required=True)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--fps", type=int, default=8,
                        help="Animation speed (default: 8)")
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(args.embeddings_dir))
    if args.clips_dir is None:
        args.clips_dir = os.path.join(base, "anomaly_clips")
    if args.heatmaps_dir is None:
        args.heatmaps_dir = os.path.join(base, "heatmaps")

    os.makedirs(os.path.dirname(os.path.abspath(args.labels_file)), exist_ok=True)

    all_files = sorted(
        f for f in os.listdir(args.embeddings_dir)
        if f.startswith(args.camera_id) and f.endswith("_emb.pt")
    )

    if not all_files:
        print(f"No embeddings found for camera '{args.camera_id}' in {args.embeddings_dir}")
        return

    labelled = load_existing_labels(args.labels_file)
    pending = [f for f in all_files if f not in labelled]

    print(f"\nCamera  : {args.camera_id}")
    print(f"Total   : {len(all_files)}  |  Labelled: {len(labelled)}  |  Pending: {len(pending)}")
    print("Each clip opens as an animated GIF (red = anomaly region, loops until you label it).")
    print("Controls: [t] true anomaly  [f] false positive  [s] skip  [q] quit\n")

    if not pending:
        print("All clips already labelled.")
        return

    gif_dir = os.path.join(os.path.dirname(os.path.abspath(args.embeddings_dir)),
                           "review_gifs")
    os.makedirs(gif_dir, exist_ok=True)
    print(f"GIFs saved to: {gif_dir}")
    print("Open them in VSCode Explorer (click to preview — animated GIFs play inline).\n")

    saved = 0

    with open(args.labels_file, "a") as out:
        for i, clip_file in enumerate(pending):
            number = clip_number(clip_file)
            clip_path = find_clip_tensor(args.clips_dir, args.camera_id, number)
            heatmap_path = os.path.join(args.heatmaps_dir, f"{args.camera_id}_clip{number}.png")

            print(f"[{i+1}/{len(pending)}] clip {number}")

            if clip_path:
                gif_path = os.path.join(gif_dir, f"clip_{number}.gif")
                try:
                    render_gif(clip_path, heatmap_path, gif_path, fps=args.fps)
                    print(f"  → {gif_path}")
                except Exception as e:
                    print(f"  (could not render GIF: {e})")
            else:
                print(f"  (clip tensor not found in {args.clips_dir} — label by number only)")

            while True:
                key = input("  Label [t/f/s/q]: ").strip().lower()
                if key in ("t", "f", "s", "q"):
                    break
                print("  Invalid — enter t / f / s / q")

            if key == "q":
                print(f"\nQuitting. {saved} labels saved this session.")
                return

            if key == "s":
                print("  Skipped.\n")
                continue

            label = 1 if key == "t" else 0
            record = {"clip": clip_file, "label": label, "camera_id": args.camera_id}
            out.write(json.dumps(record) + "\n")
            out.flush()
            saved += 1
            print(f"  Saved as {'true_anomaly' if label else 'false_positive'}.\n")

    print(f"\nDone. {saved} labels saved to {args.labels_file}")
    print("Run training/train_feedback.py to train the Stage 2 classifier.")


if __name__ == "__main__":
    main()
