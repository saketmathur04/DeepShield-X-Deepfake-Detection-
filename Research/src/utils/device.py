"""
DeepShield-X — Device Utilities

Automatic device detection: CUDA → MPS → CPU.
"""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def get_device(preferred: Optional[str] = None) -> torch.device:
    """
    Auto-detect the best available device.

    Priority: CUDA → MPS → CPU (unless overridden).

    Args:
        preferred: Force a specific device ("cuda", "mps", "cpu").

    Returns:
        torch.device instance.
    """
    if preferred and preferred != "auto":
        device = torch.device(preferred)
        logger.info(f"Using requested device: {device}")
        return device

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Using CUDA device: {gpu_name}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")

    return device


def device_info(device: torch.device) -> dict:
    """Get detailed info about the current device."""
    info = {"device": str(device), "type": device.type}

    if device.type == "cuda":
        info["name"] = torch.cuda.get_device_name(device)
        props = torch.cuda.get_device_properties(device)
        info["total_memory_gb"] = round(props.total_mem / 1e9, 2)
        info["compute_capability"] = f"{props.major}.{props.minor}"

    return info
