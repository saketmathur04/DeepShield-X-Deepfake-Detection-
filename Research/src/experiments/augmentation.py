"""DeepShield-X — Augmentation Experiment (Experiment 2)"""

from typing import Dict


def get_augmentation_configs() -> Dict[str, Dict]:
    """Return configs for no-augmentation vs augmented training."""
    return {
        "no_augmentation": {"augmentation": {"enabled": False}},
        "with_augmentation": {"augmentation": {"enabled": True}},
    }
