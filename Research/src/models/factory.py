"""
DeepShield-X — Model Factory

Unified model creation interface.
Usage:
    model = create_model("resnet50", config)
    model(images)  # works for any model

Switching models requires only a config change — the training pipeline
never needs to know which architecture is being used.
"""

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from src.models.cnn import create_custom_cnn
from src.models.resnet import create_resnet50
from src.models.efficientnet import create_efficientnet
from src.models.mobilenet import create_mobilenet
from src.models.xception import create_xception
from src.models.vit import create_vit

logger = logging.getLogger(__name__)

# Model registry — maps config name to factory function
MODEL_REGISTRY = {
    "custom_cnn": create_custom_cnn,
    "resnet50": create_resnet50,
    "efficientnet": create_efficientnet,
    "mobilenet": create_mobilenet,
    "xception": create_xception,
    "vit": create_vit,
}


def create_model(
    name: str,
    num_classes: int = 1,
    pretrained: bool = True,
    dropout: float = 0.5,
    finetune_strategy: str = "full",
    unfreeze_layers: int = 2,
    **kwargs,
) -> nn.Module:
    """
    Create a model by name using the factory registry.

    Args:
        name: Model name (must be in MODEL_REGISTRY).
        num_classes: Number of output classes (1 for binary sigmoid).
        pretrained: Whether to use pretrained weights.
        dropout: Dropout rate.
        finetune_strategy: "frozen", "partial", or "full".
        unfreeze_layers: Number of layers to unfreeze (partial only).
        **kwargs: Additional model-specific arguments.

    Returns:
        nn.Module instance.

    Raises:
        ValueError: If model name is not registered.
    """
    if name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{name}'. Available: {available}")

    factory_fn = MODEL_REGISTRY[name]

    model = factory_fn(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        finetune_strategy=finetune_strategy,
        unfreeze_layers=unfreeze_layers,
        **kwargs,
    )

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Created model: {name} | "
        f"Total params: {total_params:,} | "
        f"Trainable: {trainable_params:,} | "
        f"Strategy: {finetune_strategy}"
    )

    return model


def get_model_info(model: nn.Module, name: str = "unknown") -> Dict[str, Any]:
    """
    Get model information for experiment tracking.

    Args:
        model: The model instance.
        name: Model name string.

    Returns:
        Dict with parameter counts, model size, etc.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Estimate model size in MB
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / (1024 * 1024)

    return {
        "name": name,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": total_params - trainable_params,
        "model_size_mb": round(model_size_mb, 2),
    }


def list_available_models():
    """Return list of registered model names."""
    return list(MODEL_REGISTRY.keys())
