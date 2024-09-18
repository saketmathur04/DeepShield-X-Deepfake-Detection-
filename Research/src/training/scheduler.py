"""
DeepShield-X — Learning Rate Scheduler Factory

Configurable LR scheduling from config.
"""

import torch
from typing import Any, Dict, Optional


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str = "cosine",
    epochs: int = 20,
    scheduler_params: Optional[Dict[str, Any]] = None,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create a learning rate scheduler by name.

    Args:
        optimizer: The optimizer to schedule.
        scheduler_name: "cosine", "step", "plateau", or "none".
        epochs: Total number of training epochs.
        scheduler_params: Additional scheduler parameters.

    Returns:
        Scheduler instance, or None if "none".
    """
    params = scheduler_params or {}

    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-7
        )
    elif scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=params.get("step_size", 7),
            gamma=params.get("gamma", 0.1),
        )
    elif scheduler_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=params.get("patience", 3),
            factor=params.get("factor", 0.5),
            min_lr=1e-7,
        )
    elif scheduler_name == "none":
        return None
    else:
        raise ValueError(
            f"Unknown scheduler: {scheduler_name}. Available: cosine, step, plateau, none"
        )
