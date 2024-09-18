"""
    
DeepShield-X — Optimizer Factory

Configurable optimizer creation from config.
"""

import torch
import torch.nn as nn
from typing import Optional


def create_optimizer(
    model: nn.Module,
    optimizer_name: str = "adamw",
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    momentum: float = 0.9,
) -> torch.optim.Optimizer:
    """
    Create an optimizer by name.

    Args:
        model: Model whose parameters to optimize.
        optimizer_name: "adam", "adamw", or "sgd".
        learning_rate: Learning rate.
        weight_decay: Weight decay (L2 regularization).
        momentum: Momentum (SGD only).

    Returns:
        Optimizer instance.
    """
    # Only optimize parameters that require gradients
    params = filter(lambda p: p.requires_grad, model.parameters())

    if optimizer_name == "adam":
        return torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        return torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        return torch.optim.SGD(
            params, lr=learning_rate, weight_decay=weight_decay, momentum=momentum
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}. Available: adam, adamw, sgd")
