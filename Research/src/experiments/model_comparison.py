"""DeepShield-X — Model Comparison Experiment (Experiment 1)"""

# Implementation is in scripts/run_experiments.py → run_model_comparison()
# This module provides helpers for standalone usage.

from typing import Dict, List
from src.models.factory import list_available_models


def get_comparison_models(config: Dict = None) -> List[str]:
    """Get list of models to compare."""
    if config and "models" in config:
        return config["models"]
    return list_available_models()
