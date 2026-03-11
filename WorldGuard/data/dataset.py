"""ClipDataset — loads pre-extracted .pt clip tensors and splits them into
context frames (first 75%) and target frames (last 25%).

Expected clip format: torch.Tensor of shape (clip_frames, 3, H, W), float32 in [0, 1].
Produced by data/extract_clips.py.

Usage:
    from data.dataset import ClipDataset

    dataset = ClipDataset(clip_dir="data/train", config=cfg, augment=True)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    for context, target in loader:
        # context: (B, 12, 3, 224, 224)
        # target:  (B, 4,  3, 224, 224)
        ...
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset

from data.augmentations import ConsistentAugment, NormalizeVideo


class ClipDataset(Dataset):
    """Dataset of fixed-length video clips for self-supervised JEPA training.

    Returns (context_frames, target_frames) — no labels, ever.

    Args:
        clip_dir: Directory containing .pt clip files (output of extract_clips.py).
        config: Loaded YAML config dict.
        augment: Apply random crop + flip augmentation. Use True for train, False for val.
    """

    def __init__(self, clip_dir: str, config: dict, augment: bool = True) -> None:
        self.clip_dir = Path(clip_dir)
        self.files = sorted(self.clip_dir.glob("*.pt"))

        if not self.files:
            raise FileNotFoundError(
                f"No .pt clip files found in {clip_dir}. "
                "Run data/extract_clips.py first."
            )

        cfg = config["data"]
        self.clip_frames: int = cfg["clip_frames"]
        self.context_frames: int = cfg["context_frames"]
        self.target_frames: int = cfg["target_frames"]

        if self.context_frames + self.target_frames != self.clip_frames:
            raise ValueError(
                f"context_frames ({self.context_frames}) + target_frames "
                f"({self.target_frames}) must equal clip_frames ({self.clip_frames})"
            )

        self.augment = augment
        self.aug = ConsistentAugment(frame_size=cfg["frame_size"]) if augment else None
        self.normalize = NormalizeVideo()

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # weights_only=True: safe loading, prevents arbitrary pickle execution
        clip: torch.Tensor = torch.load(self.files[idx], weights_only=True)
        # clip shape: (clip_frames, 3, H, W)

        if self.aug is not None:
            clip = self.aug(clip)

        clip = self.normalize(clip)

        context = clip[: self.context_frames]  # (12, 3, 224, 224)
        target = clip[self.context_frames :]   # (4,  3, 224, 224)

        return context, target
