"""Extract 16-frame clips from a video file or a directory of image frames.

Clips are saved as float32 tensors of shape (clip_frames, 3, H, W) in [0, 1].
Normalization is intentionally NOT applied here — that happens in dataset.py.

Usage:
    # From a video file (MP4, AVI, etc.):
    python data/extract_clips.py \
        --video /path/to/cam01.mp4 \
        --output-dir data/train \
        --config configs/train_default.yaml \
        --camera-id cam01

    # From a directory of image frames (e.g. UCSD Ped2 .tif sequences):
    python data/extract_clips.py \
        --frames-dir data/UCSDpeds2/UCSDped2/Train/Train001 \
        --output-dir data/train \
        --camera-id ucsd

    # Override stride from config:
    python data/extract_clips.py --video cam01.mp4 --output-dir data/val \
        --camera-id cam01 --stride 8
"""

import argparse
from pathlib import Path

import av
import torch
import torchvision.transforms.functional as TF
import yaml
from PIL import Image


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _save_clips_from_buffer(
    frame_buffer: list,
    clip_frames: int,
    stride: int,
    out_dir: Path,
    camera_id: str,
    clip_idx: int,
) -> tuple[list, int]:
    """Save a clip when buffer is full; slide window. Returns (updated_buffer, clip_idx)."""
    if len(frame_buffer) == clip_frames:
        clip = torch.stack(frame_buffer)
        fname = f"{camera_id}_clip_{clip_idx:06d}.pt"
        torch.save(clip, out_dir / fname)
        clip_idx += 1
        if clip_idx % 100 == 0:
            print(f"  Saved {clip_idx} clips...")
        frame_buffer = frame_buffer[stride:]
    return frame_buffer, clip_idx


def extract_clips(
    video_path: str,
    output_dir: str,
    config: dict,
    camera_id: str,
    stride_override: int | None = None,
) -> int:
    """Extract sliding-window clips from a video file and save as .pt tensors.

    Args:
        video_path: Path to the source video file.
        output_dir: Directory where clip tensors will be saved.
        config: Loaded YAML config dict.
        camera_id: Used as a filename prefix (e.g. "cam01").
        stride_override: If set, overrides data.stride from config.

    Returns:
        Number of clips saved.
    """
    cfg = config["data"]
    clip_frames: int = cfg["clip_frames"]
    frame_size: int = cfg["frame_size"]
    stride: int = stride_override if stride_override is not None else cfg["stride"]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    container = av.open(str(video_path))
    video_stream = container.streams.video[0]

    total_frames = video_stream.frames  # may be 0 if container doesn't report it
    if total_frames:
        print(f"Video: {video_path}  |  ~{total_frames} frames  |  stride={stride}")
    else:
        print(f"Video: {video_path}  |  stride={stride}")

    frame_buffer: list[torch.Tensor] = []
    clip_idx = 0

    for frame in container.decode(video_stream):
        # Decode → PIL → resize → tensor (C, H, W) in [0, 1]
        img: Image.Image = frame.to_image().convert("RGB")
        img = img.resize((frame_size, frame_size), resample=Image.BILINEAR)
        tensor = TF.to_tensor(img)  # float32, (3, H, W), [0, 1]
        frame_buffer.append(tensor)
        frame_buffer, clip_idx = _save_clips_from_buffer(
            frame_buffer, clip_frames, stride, out_dir, camera_id, clip_idx
        )

    container.close()

    if clip_idx == 0:
        print(f"WARNING: No clips extracted. Video may be shorter than {clip_frames} frames.")
    else:
        print(f"Done. Saved {clip_idx} clips → {out_dir}")

    return clip_idx


def extract_clips_from_frames(
    frames_dir: str,
    output_dir: str,
    config: dict,
    camera_id: str,
    stride_override: int | None = None,
) -> int:
    """Extract sliding-window clips from a directory of image frames.

    Supports .tif/.tiff, .png, .jpg/.jpeg files. Frames are sorted alphabetically.
    Grayscale images are automatically converted to RGB.

    Args:
        frames_dir: Directory containing sorted image frames.
        output_dir: Directory where clip tensors will be saved.
        config: Loaded YAML config dict.
        camera_id: Used as a filename prefix (e.g. "ucsd").
        stride_override: If set, overrides data.stride from config.

    Returns:
        Number of clips saved.
    """
    cfg = config["data"]
    clip_frames: int = cfg["clip_frames"]
    frame_size: int = cfg["frame_size"]
    stride: int = stride_override if stride_override is not None else cfg["stride"]

    frames_path = Path(frames_dir)
    image_extensions = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
    frame_files = sorted(
        f for f in frames_path.iterdir()
        if f.suffix.lower() in image_extensions
    )

    if not frame_files:
        print(f"WARNING: No image files found in {frames_dir}")
        return 0

    print(f"Frames dir: {frames_dir}  |  {len(frame_files)} frames  |  stride={stride}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_buffer: list[torch.Tensor] = []
    clip_idx = 0

    for fpath in frame_files:
        img = Image.open(fpath).convert("RGB")  # handles grayscale → RGB
        img = img.resize((frame_size, frame_size), resample=Image.BILINEAR)
        tensor = TF.to_tensor(img)  # float32, (3, H, W), [0, 1]
        frame_buffer.append(tensor)
        frame_buffer, clip_idx = _save_clips_from_buffer(
            frame_buffer, clip_frames, stride, out_dir, camera_id, clip_idx
        )

    if clip_idx == 0:
        print(f"WARNING: No clips extracted. Directory may have fewer than {clip_frames} frames.")
    else:
        print(f"Done. Saved {clip_idx} clips → {out_dir}")

    return clip_idx


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract fixed-length clips from a video file or frame directory"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", help="Path to input video file (MP4, AVI, etc.)")
    source.add_argument(
        "--frames-dir",
        help="Path to directory of sorted image frames (tif/png/jpg) — e.g. UCSD Ped2 sequences",
    )
    parser.add_argument("--output-dir", required=True, help="Directory to save clip tensors")
    parser.add_argument(
        "--config", default="configs/train_default.yaml", help="Path to YAML config"
    )
    parser.add_argument(
        "--camera-id", default="cam00", help="Camera identifier prefix for output filenames"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Frame stride for sliding window (overrides config value)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.video:
        extract_clips(args.video, args.output_dir, config, args.camera_id, args.stride)
    else:
        extract_clips_from_frames(
            args.frames_dir, args.output_dir, config, args.camera_id, args.stride
        )


if __name__ == "__main__":
    main()
