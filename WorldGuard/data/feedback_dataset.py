"""Dataset for Stage 2 feedback classifier training.

Loads (z_pred embedding, label) pairs from a JSON labels file
and a directory of saved embedding .pt files.

Labels file format (data/feedback/labels.json) — one JSON object per line:
    {"clip": "ucsd_clip000042_emb.pt", "label": 1, "camera_id": "ucsd"}
    {"clip": "ucsd_clip000071_emb.pt", "label": 0, "camera_id": "ucsd"}

label=1 → true anomaly, label=0 → false positive
"""

import json
import os

import torch
from torch.utils.data import Dataset


class FeedbackDataset(Dataset):
    """Dataset of labeled z_pred embeddings for Stage 2 training.

    Args:
        labels_file: Path to JSON labels file (one record per line).
        embeddings_dir: Directory containing .pt embedding files.
        camera_id: Only load labels for this camera (or None for all).
    """

    def __init__(
        self,
        labels_file: str,
        embeddings_dir: str,
        camera_id: str | None = None,
    ) -> None:
        self.embeddings_dir = embeddings_dir
        self.records = []

        with open(labels_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if camera_id and record.get("camera_id") != camera_id:
                    continue
                emb_path = os.path.join(embeddings_dir, record["clip"])
                if not os.path.isfile(emb_path):
                    continue
                self.records.append(record)

        if not self.records:
            raise ValueError(
                f"No labelled embeddings found in {labels_file} "
                f"for camera_id={camera_id!r}. "
                f"Run training/review_anomalies.py first."
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[idx]
        emb_path = os.path.join(self.embeddings_dir, record["clip"])
        z_pred = torch.load(emb_path, map_location="cpu", weights_only=True)
        label = torch.tensor(float(record["label"]), dtype=torch.float32)
        return z_pred, label
