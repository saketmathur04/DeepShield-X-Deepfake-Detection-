"""
DeepShield-X — Checkpoint Utilities

Save/load model checkpoints with full training state and config snapshots.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    config: Dict[str, Any],
    path: str,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    is_best: bool = False,
) -> None:
    """
    Save a training checkpoint.

    Args:
        model: The model to save.
        optimizer: Current optimizer state.
        epoch: Current epoch number.
        metrics: Dict of metric values at this checkpoint.
        config: Full config dict for reproducibility.
        path: Save path for the checkpoint.
        scheduler: Optional LR scheduler state.
        scaler: Optional GradScaler state (mixed precision).
        is_best: Whether this is the best model so far.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": config,
        "is_best": is_best,
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved: {save_path} (epoch={epoch}, best={is_best})")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load a training checkpoint.

    Args:
        path: Path to the checkpoint file.
        model: Model to load weights into.
        optimizer: Optional optimizer to restore state.
        scheduler: Optional scheduler to restore state.
        scaler: Optional GradScaler to restore state.
        device: Device to map checkpoint to.

    Returns:
        Dict with epoch, metrics, and config from checkpoint.
    """
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    map_location = device if device else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Model weights loaded from: {checkpoint_path}")

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "metrics": checkpoint.get("metrics", {}),
        "config": checkpoint.get("config", {}),
        "is_best": checkpoint.get("is_best", False),
    }


def save_training_history(history: Dict, path: str) -> None:
    """Save training history (loss/metric curves) to JSON."""
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, default=str)
    logger.info(f"Training history saved: {save_path}")


def load_training_history(path: str) -> Dict:
    """Load training history from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
