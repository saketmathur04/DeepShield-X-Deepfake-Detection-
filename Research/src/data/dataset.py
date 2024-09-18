"""
DeepShield-X — Dataset Abstraction

Provides DeepFakeDataset (for real data) and SyntheticDataset (for smoke tests).
Both use the same interface: returns (image_tensor, label, metadata_dict).
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class DeepFakeDataset(Dataset):
    """
    Dataset for deepfake detection.

    Loads images and labels from a metadata CSV with columns:
        - image_path: path to the face image
        - label: 0 (REAL) or 1 (FAKE)
        - source_id: video/source identifier (for leakage prevention)
    
    Args:
        metadata_csv: Path to CSV with image_path, label, source_id columns.
        root_dir: Root directory prepended to relative image paths.
        transform: Optional image transform (Albumentations or torchvision).
        image_size: Target image size (height, width).
    """

    LABEL_MAP = {0: "REAL", 1: "FAKE"}

    def __init__(
        self,
        metadata_csv: str,
        root_dir: str = ".",
        transform: Optional[Callable] = None,
        image_size: int = 224,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_size = image_size

        self.metadata = pd.read_csv(metadata_csv)
        required_cols = {"image_path", "label"}
        if not required_cols.issubset(self.metadata.columns):
            raise ValueError(f"Metadata CSV must contain columns: {required_cols}")

        # Validate labels are binary
        unique_labels = self.metadata["label"].unique()
        if not set(unique_labels).issubset({0, 1}):
            raise ValueError(f"Labels must be 0 or 1, got: {unique_labels}")

        logger.info(
            f"Loaded dataset: {len(self.metadata)} images "
            f"(REAL={sum(self.metadata['label']==0)}, FAKE={sum(self.metadata['label']==1)})"
        )

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        row = self.metadata.iloc[idx]
        
        # Flexibly handle absolute or relative image paths
        img_p = Path(row["image_path"])
        # Strip leading ./ or normalize
        parts = img_p.parts
        if parts and parts[0] in (".", "data"):
            stripped_p = Path(*[p for p in parts if p != "."])
            if stripped_p.parts and stripped_p.parts[0] == "data":
                stripped_p = Path(*stripped_p.parts[1:])
        else:
            stripped_p = img_p

        if img_p.is_absolute() and img_p.exists():
            image_path = img_p
        elif (self.root_dir / img_p).exists():
            image_path = self.root_dir / img_p
        elif (self.root_dir / stripped_p).exists():
            image_path = self.root_dir / stripped_p
        elif (self.root_dir.parent / img_p).exists():
            image_path = self.root_dir.parent / img_p
        else:
            image_path = self.root_dir / stripped_p

        label = int(row["label"])

        # Load and convert to RGB
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        image = np.array(image)

        # Apply transform
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        else:
            # Default: normalize to [0,1] and convert to tensor
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        metadata = {
            "image_path": str(image_path),
            "source_id": row.get("source_id", "unknown"),
            "label_name": self.LABEL_MAP[label],
        }

        return image, label, metadata

    @property
    def num_real(self) -> int:
        return int((self.metadata["label"] == 0).sum())

    @property
    def num_fake(self) -> int:
        return int((self.metadata["label"] == 1).sum())

    @property
    def class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for balanced training."""
        counts = self.metadata["label"].value_counts().sort_index()
        total = len(self.metadata)
        weights = total / (len(counts) * counts.values)
        return torch.FloatTensor(weights)


class SyntheticDataset(Dataset):
    """
    Synthetic dataset for smoke testing.

    Generates random RGB images with random binary labels.
    No disk I/O required — useful for verifying the full pipeline.

    Args:
        num_samples: Number of synthetic samples.
        image_size: Image dimensions (height=width).
        num_classes: Always 1 for binary.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        num_samples: int = 100,
        image_size: int = 64,
        num_classes: int = 1,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.rng = np.random.RandomState(seed)

        # Pre-generate labels (balanced)
        self.labels = np.array([i % 2 for i in range(num_samples)])
        self.rng.shuffle(self.labels)

        logger.info(
            f"Synthetic dataset: {num_samples} samples at {image_size}x{image_size} "
            f"(REAL={sum(self.labels==0)}, FAKE={sum(self.labels==1)})"
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        label = int(self.labels[idx])

        # Generate a distinct pattern for REAL vs FAKE so models can learn something
        rng = np.random.RandomState(idx)
        image = rng.rand(3, self.image_size, self.image_size).astype(np.float32)

        if label == 1:  # FAKE — add a subtle grid pattern
            for i in range(0, self.image_size, 4):
                image[:, i, :] *= 0.7
                image[:, :, i] *= 0.7

        image = torch.from_numpy(image)

        metadata = {
            "image_path": f"synthetic_{idx}.jpg",
            "source_id": f"video_{idx // 10}",
            "label_name": "FAKE" if label == 1 else "REAL",
        }

        return image, label, metadata


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """
    Create a DataLoader with standard settings.

    Args:
        dataset: PyTorch Dataset instance.
        batch_size: Batch size.
        shuffle: Whether to shuffle data.
        num_workers: Number of parallel workers.
        pin_memory: Pin memory for CUDA transfers.
        drop_last: Drop the last incomplete batch.

    Returns:
        DataLoader instance.
    """
    # Custom collate to handle metadata dicts
    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch], dtype=torch.float32)
        metadata = [item[2] for item in batch]
        return images, labels, metadata

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )


# Alias for backward compatibility
create_data_loader = create_dataloader
