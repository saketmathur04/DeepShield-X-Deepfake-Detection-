"""
DeepShield-X — Reproducibility Utilities

Set deterministic seeds and record environment information.
"""

import os
import platform
import random
import sys
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Multi-GPU

    # Deterministic algorithms (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def get_environment_info() -> Dict[str, Any]:
    """
    Record runtime environment for reproducibility.
    
    Returns:
        Dict containing Python, PyTorch, CUDA, and system information.
    """
    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn_version": str(torch.backends.cudnn.version()) if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    if torch.cuda.is_available():
        info["gpu_names"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
        info["gpu_memory"] = [
            f"{torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB"
            for i in range(torch.cuda.device_count())
        ]

    # MPS (Apple Silicon)
    if hasattr(torch.backends, "mps"):
        info["mps_available"] = torch.backends.mps.is_available()

    return info
