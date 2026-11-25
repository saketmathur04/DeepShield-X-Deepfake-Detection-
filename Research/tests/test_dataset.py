"""
DeepShield-X — Dataset Tests

Verify dataset loading, synthetic data, and splitting.
"""

import pytest
import numpy as np
import pandas as pd
import torch

from src.data.dataset import SyntheticDataset, create_dataloader
from src.data.splitting import source_aware_split, check_leakage


class TestSyntheticDataset:
    """Test the synthetic smoke-test dataset."""

    def test_creation(self):
        ds = SyntheticDataset(num_samples=20, image_size=32, seed=42)
        assert len(ds) == 20

    def test_getitem_returns_correct_types(self):
        ds = SyntheticDataset(num_samples=10, image_size=32)
        image, label, metadata = ds[0]

        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 32, 32)
        assert isinstance(label, int)
        assert label in (0, 1)
        assert isinstance(metadata, dict)
        assert "source_id" in metadata

    def test_balanced_labels(self):
        ds = SyntheticDataset(num_samples=20, image_size=32)
        labels = [ds[i][1] for i in range(len(ds))]
        assert sum(labels) == 10  # Half real, half fake

    def test_dataloader(self):
        ds = SyntheticDataset(num_samples=16, image_size=32)
        loader = create_dataloader(ds, batch_size=4, shuffle=False, num_workers=0)

        batch_images, batch_labels, batch_metadata = next(iter(loader))
        assert batch_images.shape == (4, 3, 32, 32)
        assert batch_labels.shape == (4,)
        assert len(batch_metadata) == 4


class TestSourceAwareSplitting:
    """Test source-level data splitting."""

    def test_basic_split(self):
        data = []
        for vid in range(10):
            for frame in range(5):
                data.append({
                    "image_path": f"video_{vid}/frame_{frame}.jpg",
                    "label": vid % 2,
                    "source_id": f"video_{vid}",
                })

        df = pd.DataFrame(data)
        train, val, test = source_aware_split(df, 0.6, 0.2, 0.2, seed=42)

        # Check no leakage
        train_sources = set(train["source_id"])
        val_sources = set(val["source_id"])
        test_sources = set(test["source_id"])

        assert len(train_sources & val_sources) == 0
        assert len(train_sources & test_sources) == 0
        assert len(val_sources & test_sources) == 0

    def test_all_frames_same_split(self):
        """All frames from a video must be in the same split."""
        data = []
        for vid in range(10):
            for frame in range(10):
                data.append({
                    "image_path": f"v{vid}/f{frame}.jpg",
                    "label": vid % 2,
                    "source_id": f"v{vid}",
                })

        df = pd.DataFrame(data)
        train, val, test = source_aware_split(df, 0.6, 0.2, 0.2, seed=42)

        for split_df in [train, val, test]:
            for source_id in split_df["source_id"].unique():
                # Count frames for this source in this split
                count_in_split = len(split_df[split_df["source_id"] == source_id])
                # Should match total frames for this source
                count_total = len(df[df["source_id"] == source_id])
                assert count_in_split == count_total

    def test_leakage_detection_raises(self):
        """check_leakage should raise on overlapping sources."""
        train = pd.DataFrame({"source_id": ["v1", "v2"], "label": [0, 1]})
        val = pd.DataFrame({"source_id": ["v2", "v3"], "label": [0, 1]})
        test = pd.DataFrame({"source_id": ["v4"], "label": [0]})

        with pytest.raises(ValueError, match="DATA LEAKAGE"):
            check_leakage(train, val, test)


class TestUnionFindGrouping:
    """Test the Union-Find connected component grouping for identity leakage prevention."""

    def test_build_identity_groups_basic(self):
        from src.data.splitting import build_identity_groups
        df = pd.DataFrame({
            "video_path": ["001.mp4", "001_002.mp4", "002_001.mp4", "003.mp4"],
            "identities": [["001"], ["001", "002"], ["002", "001"], ["003"]],
            "label": [0, 1, 1, 0]
        })
        
        grouped = build_identity_groups(df)
        assert "group_id" in grouped.columns
        
        # 001 and 002 should be in the same group
        g1 = grouped.iloc[0]["group_id"]
        g2 = grouped.iloc[1]["group_id"]
        g3 = grouped.iloc[2]["group_id"]
        assert g1 == g2 == g3
        
        # 003 should be in its own group
        g4 = grouped.iloc[3]["group_id"]
        assert g1 != g4

    def test_build_identity_groups_chain(self):
        from src.data.splitting import build_identity_groups
        df = pd.DataFrame({
            "identities": [["001", "002"], ["002", "003"], ["003", "004"]],
            "label": [1, 1, 1]
        })
        grouped = build_identity_groups(df)
        
        # All should be collapsed into exactly one group
        groups = grouped["group_id"].unique()
        assert len(groups) == 1

    def test_deepfakedetection_labels(self):
        # Emulate the parsing from our prepare script
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from scripts.prepare_ff_c23_dataset import get_identities
        
        # Original
        assert get_identities("001.mp4", "original") == ["001"]
        # Manipulation
        assert get_identities("001_002.mp4", "Deepfakes") == ["001", "002"]
        # DeepFakeDetection (often different format)
        assert get_identities("actor1_actor2_123.mp4", "DeepFakeDetection") == ["actor1", "actor2", "123"]
        assert get_identities("complex_video.mp4", "DeepFakeDetection") == ["complex_video"]
