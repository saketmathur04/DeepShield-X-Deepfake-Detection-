"""
DeepShield-X — ROC and PR Curve Visualization
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc

logger = logging.getLogger(__name__)


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float,
    model_name: str = "Model",
    save_path: Optional[str] = None,
    figsize: tuple = (8, 6),
) -> None:
    """Plot a single ROC curve."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"ROC curve saved: {save_path}")
    plt.close()


def plot_roc_comparison(
    curves: Dict[str, Dict[str, np.ndarray]],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8),
) -> None:
    """
    Plot multiple ROC curves for model comparison.

    Args:
        curves: Dict mapping model_name → {"fpr": array, "tpr": array, "auc": float}.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for model_name, data in curves.items():
        ax.plot(
            data["fpr"], data["tpr"], lw=2,
            label=f"{model_name} (AUC = {data['auc']:.4f})"
        )

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_pr_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    pr_auc: float,
    model_name: str = "Model",
    save_path: Optional[str] = None,
    figsize: tuple = (8, 6),
) -> None:
    """Plot a single Precision-Recall curve."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(recall, precision, lw=2, label=f"{model_name} (AP = {pr_auc:.4f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
