"""
DeepShield-X — Centralized Visualization System

Publication-quality plots with consistent styling.
Every plot auto-saves to the configured output directory.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Global style settings for publication-quality plots
STYLE_CONFIG = {
    "figure.figsize": (10, 6),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
}

# Color palette
COLORS = [
    "#2196F3",  # Blue
    "#FF5722",  # Deep Orange
    "#4CAF50",  # Green
    "#9C27B0",  # Purple
    "#FF9800",  # Orange
    "#00BCD4",  # Cyan
    "#E91E63",  # Pink
    "#795548",  # Brown
]


def setup_plot_style():
    """Apply consistent plot styling."""
    plt.rcParams.update(STYLE_CONFIG)
    sns.set_palette(COLORS)
    sns.set_style("whitegrid")


def plot_metric_comparison(
    results: Dict[str, float],
    metric_name: str,
    title: str = "",
    save_path: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: tuple = (10, 6),
) -> None:
    """
    Bar chart comparing a single metric across models.

    Args:
        results: Dict mapping model_name → metric_value.
        metric_name: Name of the metric being compared.
        title: Plot title.
        save_path: Path to save the figure.
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=figsize)

    models = list(results.keys())
    values = list(results.values())
    colors = COLORS[:len(models)]

    bars = ax.bar(models, values, color=colors, edgecolor="white", linewidth=0.5)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    ax.set_ylabel(ylabel or metric_name.replace("_", " ").title(), fontsize=12)
    ax.set_title(title or f"{metric_name.replace('_', ' ').title()} Comparison", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.15 if values else 1)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Plot saved: {save_path}")
    plt.close()


def plot_training_curves(
    history: Dict[str, List[float]],
    model_name: str = "Model",
    save_path: Optional[str] = None,
    figsize: tuple = (14, 5),
) -> None:
    """
    Plot training and validation loss/accuracy curves side by side.

    Args:
        history: Dict with train_loss, val_loss, train_acc, val_acc lists.
        model_name: Model name for title.
        save_path: Path to save the figure.
    """
    setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    epochs = range(1, len(history.get("train_loss", [])) + 1)

    # Loss curves
    if "train_loss" in history:
        ax1.plot(epochs, history["train_loss"], "b-", lw=2, label="Train Loss")
    if "val_loss" in history:
        ax1.plot(epochs, history["val_loss"], "r-", lw=2, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{model_name} — Loss", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy curves
    if "train_acc" in history:
        ax2.plot(epochs, history["train_acc"], "b-", lw=2, label="Train Acc")
    if "val_acc" in history:
        ax2.plot(epochs, history["val_acc"], "r-", lw=2, label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"{model_name} — Accuracy", fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_robustness_curve(
    x_values: List,
    y_values: List[float],
    xlabel: str,
    ylabel: str = "F1 Score",
    title: str = "Robustness Analysis",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
) -> None:
    """Plot a line chart for robustness experiments (e.g., JPEG quality vs F1)."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(x_values, y_values, "o-", lw=2, markersize=8, color=COLORS[0])

    for x, y in zip(x_values, y_values):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_multi_metric_comparison(
    results: Dict[str, Dict[str, float]],
    metrics: List[str],
    title: str = "Model Comparison",
    save_path: Optional[str] = None,
    figsize: tuple = (14, 6),
) -> None:
    """
    Grouped bar chart comparing multiple metrics across models.

    Args:
        results: Dict mapping model_name → {metric_name: value}.
        metrics: List of metric names to compare.
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=figsize)

    models = list(results.keys())
    x = np.arange(len(models))
    width = 0.8 / len(metrics)

    for i, metric in enumerate(metrics):
        values = [results[m].get(metric, 0) for m in models]
        offset = (i - len(metrics) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=metric.replace("_", " ").title(),
                      color=COLORS[i % len(COLORS)])

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_confidence_distribution(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
) -> None:
    """Plot confidence distribution for correct vs incorrect predictions."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=figsize)

    y_pred = (y_prob >= 0.5).astype(int)
    correct = y_prob[y_true == y_pred]
    incorrect = y_prob[y_true != y_pred]

    ax.hist(correct, bins=30, alpha=0.7, label=f"Correct (n={len(correct)})",
            color=COLORS[2], edgecolor="white")
    ax.hist(incorrect, bins=30, alpha=0.7, label=f"Incorrect (n={len(incorrect)})",
            color=COLORS[1], edgecolor="white")

    ax.set_xlabel("Confidence (Probability)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Confidence Distribution", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_efficiency_scatter(
    model_data: Dict[str, Dict[str, float]],
    x_key: str = "total_params",
    y_key: str = "f1",
    xlabel: str = "Parameters (M)",
    ylabel: str = "F1 Score",
    title: str = "Performance vs Efficiency",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8),
) -> None:
    """Scatter plot of performance vs model efficiency."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=figsize)

    for i, (model_name, data) in enumerate(model_data.items()):
        x = data.get(x_key, 0)
        y = data.get(y_key, 0)
        if x_key == "total_params":
            x = x / 1e6  # Convert to millions
        ax.scatter(x, y, s=150, color=COLORS[i % len(COLORS)],
                   edgecolors="white", linewidth=1.5, zorder=5)
        ax.annotate(model_name, (x, y), textcoords="offset points",
                    xytext=(8, 8), fontsize=10)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
