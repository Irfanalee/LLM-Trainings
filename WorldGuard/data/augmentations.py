"""Consistent spatio-temporal augmentations for video clips.

The critical invariant: random parameters (crop box, flip decision) are sampled
ONCE per clip and applied identically to every frame. Never sample per-frame —
that destroys the temporal structure the model needs to learn from.
"""

import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import RandomResizedCrop


class ConsistentAugment:
    """Random resized crop + horizontal flip applied uniformly across all frames.

    Parameters are sampled once per clip so that temporal coherence is preserved.

    Args:
        frame_size: Output spatial size (both H and W).
        scale: Lower/upper bounds for the area fraction to crop.
        ratio: Lower/upper bounds for the aspect ratio of the crop.
        hflip_prob: Probability of applying a horizontal flip.
    """

    def __init__(
        self,
        frame_size: int = 224,
        scale: tuple[float, float] = (0.8, 1.0),
        ratio: tuple[float, float] = (0.9, 1.1),
        hflip_prob: float = 0.5,
    ) -> None:
        self.frame_size = frame_size
        self.scale = scale
        self.ratio = ratio
        self.hflip_prob = hflip_prob

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            clip: (T, C, H, W) float32 in [0, 1]

        Returns:
            Augmented clip of shape (T, C, frame_size, frame_size).
        """
        T, C, H, W = clip.shape

        # Sample crop params once using torchvision's built-in sampler
        i, j, h, w = RandomResizedCrop.get_params(clip[0], self.scale, self.ratio)
        do_flip = torch.rand(1).item() < self.hflip_prob

        out_frames = []
        for t in range(T):
            frame = TF.crop(clip[t], i, j, h, w)
            frame = TF.resize(frame, [self.frame_size, self.frame_size], antialias=True)
            if do_flip:
                frame = TF.hflip(frame)
            out_frames.append(frame)

        return torch.stack(out_frames)  # (T, C, frame_size, frame_size)


class NormalizeVideo:
    """Normalize a clip with ImageNet mean and std.

    Applied after augmentation in dataset.py. The raw tensors saved by
    extract_clips.py are in [0, 1] — normalization should NOT happen there.
    """

    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            clip: (T, C, H, W) in [0, 1]

        Returns:
            Normalized clip (T, C, H, W).
        """
        return (clip - self.MEAN) / self.STD
