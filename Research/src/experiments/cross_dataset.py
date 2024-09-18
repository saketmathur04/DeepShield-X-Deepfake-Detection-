"""DeepShield-X — Cross-Dataset Generalization (Experiment 6)"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def setup_cross_dataset_evaluation(
    train_dataset_name: str = "faceforensics",
    test_dataset_name: str = "celeb_df",
    config: Dict = None,
) -> Dict[str, str]:
    """
    Configure cross-dataset evaluation.

    Train on FaceForensics++, test on Celeb-DF.
    Datasets are never mixed.
    """
    return {
        "train_dataset": train_dataset_name,
        "test_dataset": test_dataset_name,
        "description": (
            f"EXTERNAL GENERALIZATION EXPERIMENT: "
            f"Train on {train_dataset_name}, test on {test_dataset_name}. "
            f"No retraining on test dataset."
        ),
    }
