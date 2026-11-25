"""
DeepShield-X — Test Configuration & Fixtures
"""

import sys
from pathlib import Path

import pytest
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def device():
    """Get the best available device for testing."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture
def dummy_image():
    """Create a dummy image tensor [1, 3, 64, 64]."""
    return torch.randn(1, 3, 64, 64)


@pytest.fixture
def dummy_batch():
    """Create a dummy batch of images [4, 3, 64, 64]."""
    return torch.randn(4, 3, 64, 64)


@pytest.fixture
def sample_config():
    """Return a minimal config dict for testing."""
    return {
        "project": {"seed": 42, "output_dir": "./experiments"},
        "model": {"name": "custom_cnn", "num_classes": 1, "pretrained": False, "dropout": 0.5},
        "training": {
            "epochs": 2,
            "batch_size": 8,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "scheduler": "none",
            "loss": "bce",
            "label_smoothing": 0.0,
            "mixed_precision": False,
            "gradient_clip": 0,
            "early_stopping": {"enabled": False},
            "checkpoint_dir": "./experiments/checkpoints",
        },
        "dataset": {"image_size": 64, "num_workers": 0},
        "augmentation": {"enabled": False},
    }
