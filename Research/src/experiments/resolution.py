"""DeepShield-X — Resolution Experiment (Experiment 3)"""

from typing import Dict, List


def get_resolution_configs(resolutions: List[int] = None) -> Dict[str, Dict]:
    """Return configs for different resolution experiments."""
    if resolutions is None:
        resolutions = [128, 224, 384]

    return {
        f"res_{r}": {"dataset": {"image_size": r}}
        for r in resolutions
    }
