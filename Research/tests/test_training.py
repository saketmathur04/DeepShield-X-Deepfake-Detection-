"""
Tests for DeepShield-X Trainer, Checkpoint Selection, and Early Stopping.
Verifies configurable monitors (val_loss, val_roc_auc, val_pr_auc, val_f1, val_balanced_accuracy),
direction resolution (minimize vs maximize), and early stopping behavior.
"""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.trainer import EarlyStopping, Trainer, resolve_monitor_mode


class SimpleDummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 1)

    def forward(self, x):
        return self.fc(x)


def test_early_stopping_min_mode():
    """Verify EarlyStopping in minimize mode halts when score fails to decrease."""
    es = EarlyStopping(patience=3, min_delta=0.01, mode="min")
    assert not es(0.50)
    assert not es(0.40)  # improved
    assert not es(0.395) # improved < min_delta -> counter = 1
    assert not es(0.41)  # worse -> counter = 2
    assert es(0.42)      # worse -> counter = 3 >= patience -> True
    assert es.should_stop is True


def test_early_stopping_max_mode():
    """Verify EarlyStopping in maximize mode halts when score fails to increase."""
    es = EarlyStopping(patience=3, min_delta=0.01, mode="max")
    assert not es(0.80)
    assert not es(0.85)  # improved
    assert not es(0.855) # improved < min_delta -> counter = 1
    assert not es(0.84)  # worse -> counter = 2
    assert es(0.83)      # worse -> counter = 3 >= patience -> True
    assert es.should_stop is True


def test_resolve_monitor_mode_required_metrics():
    """Verify canonical resolution and directions for all required monitors."""
    required = {
        "val_loss": "min",
        "val_roc_auc": "max",
        "val_pr_auc": "max",
        "val_f1": "max",
        "val_balanced_accuracy": "max",
    }
    for metric, expected_mode in required.items():
        canonical, mode = resolve_monitor_mode(metric)
        assert canonical == metric
        assert mode == expected_mode

    # Without 'val_' prefix
    assert resolve_monitor_mode("roc_auc") == ("val_roc_auc", "max")
    assert resolve_monitor_mode("loss") == ("val_loss", "min")
    assert resolve_monitor_mode("f1") == ("val_f1", "max")

    # Explicit override
    assert resolve_monitor_mode("custom_metric", configured_mode="min") == ("val_custom_metric", "min")
    assert resolve_monitor_mode("custom_metric", configured_mode="max") == ("val_custom_metric", "max")


def test_trainer_checkpoint_selection_val_roc_auc(tmp_path):
    """Verify Trainer selects checkpoint based on highest val_roc_auc when configured."""
    model = SimpleDummyModel()
    device = torch.device("cpu")
    config = {
        "training": {
            "epochs": 2,
            "batch_size": 4,
            "learning_rate": 0.01,
            "loss": "bce",
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "checkpoint_monitor": "val_roc_auc",
            "mixed_precision": False,
            "early_stopping": {
                "enabled": True,
                "patience": 2,
                "monitor": "val_roc_auc",
                "min_delta": 0.001,
            },
        }
    }

    trainer = Trainer(model=model, device=device, config=config)
    assert trainer.monitor == "val_roc_auc"
    assert trainer.mode == "max"
    assert trainer.early_stopping is not None
    assert trainer.early_stopping.mode == "max"

    # Synthetic dataloaders with binary classes
    x_train = torch.randn(8, 4)
    y_train = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    train_ds = TensorDataset(x_train, y_train, x_train)
    train_loader = DataLoader(train_ds, batch_size=4)

    x_val = torch.randn(8, 4)
    y_val = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    val_ds = TensorDataset(x_val, y_val, x_val)
    val_loader = DataLoader(val_ds, batch_size=4)

    result = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        model_name="dummy",
        experiment_id="test_exp",
    )

    best_ckpt_path = Path(result["best_checkpoint"])
    assert best_ckpt_path.exists()
    assert result["checkpoint_monitor"] == "val_roc_auc"
    assert result["monitor_mode"] == "max"

    checkpoint = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
    assert checkpoint["is_best"] is True
    assert checkpoint["metrics"]["monitored_metric"] == "val_roc_auc"
    assert checkpoint["metrics"]["monitored_mode"] == "max"


def test_trainer_backward_compatibility_default_val_loss(tmp_path):
    """Verify Trainer defaults to val_loss minimization when no monitor is specified."""
    model = SimpleDummyModel()
    device = torch.device("cpu")
    config = {
        "training": {
            "epochs": 1,
            "batch_size": 4,
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "mixed_precision": False,
        }
    }

    trainer = Trainer(model=model, device=device, config=config)
    assert trainer.monitor == "val_loss"
    assert trainer.mode == "min"
    assert trainer.early_stopping is None
