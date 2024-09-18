"""
DeepShield-X — Confusion Matrix Visualization
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    labels: list = None,
    save_path: Optional[str] = None,
    figsize: tuple = (8, 6),
) -> None:
    """
    Plot and optionally save a confusion matrix.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        title: Plot title.
        labels: Class labels (default: ["REAL", "FAKE"]).
        save_path: Path to save the figure.
        figsize: Figure size.
    """
    if labels is None:
        labels = ["REAL", "FAKE"]

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        annot_kws={"size": 14},
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Confusion matrix saved: {save_path}")

    plt.close()
