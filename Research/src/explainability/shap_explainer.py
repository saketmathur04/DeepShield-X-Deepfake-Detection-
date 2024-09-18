"""
DeepShield-X — SHAP Explainer (Optional)

Optional SHAP-based explainability. Only used if SHAP is installed.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def is_shap_available() -> bool:
    """Check if SHAP is installed."""
    return SHAP_AVAILABLE


def generate_shap_explanation(
    model: nn.Module,
    image_tensor: torch.Tensor,
    background_data: Optional[torch.Tensor] = None,
    device: torch.device = None,
) -> Optional[Dict[str, Any]]:
    """
    Generate SHAP explanation for a single image.

    This is optional and requires SHAP to be installed.
    For most purposes, Grad-CAM is sufficient.

    Args:
        model: Trained model.
        image_tensor: Input image tensor.
        background_data: Background samples for SHAP.
        device: Compute device.

    Returns:
        Dict with SHAP values, or None if SHAP is not available.
    """
    if not SHAP_AVAILABLE:
        logger.info("SHAP not available. Install with: pip install shap")
        return None

    logger.info("SHAP explanation generation — this may take a while...")

    if device is None:
        device = torch.device("cpu")

    model.to(device)
    model.eval()

    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)

    if background_data is None:
        background_data = torch.randn_like(image_tensor[:1])

    try:
        explainer = shap.DeepExplainer(model, background_data.to(device))
        shap_values = explainer.shap_values(image_tensor.to(device))

        return {
            "shap_values": shap_values,
            "base_value": explainer.expected_value,
        }
    except Exception as e:
        logger.warning(f"SHAP explanation failed: {e}")
        return None
