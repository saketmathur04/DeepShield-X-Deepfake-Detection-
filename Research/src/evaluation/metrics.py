"""Central binary metrics engine.  REAL=0 is negative; FAKE=1 is positive."""

from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
    balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score,
    matthews_corrcoef, precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve)


def _labels(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values).astype(int).reshape(-1)
    if not np.isin(values, [0, 1]).all():
        raise ValueError("Labels must use REAL=0 and FAKE=1")
    return values


def _finite(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute all binary classification metrics.

    Convention:
        Positive class = FAKE (1)
        Negative class = REAL (0)

    Args:
        y_true: Ground truth labels (0 or 1).
        y_pred: Predicted labels (0 or 1).
        y_prob: Predicted probabilities for the positive class.
        threshold: Classification threshold.

    Returns:
        Dict with all metric values.
    """
    y_true = _labels(y_true)
    y_prob = np.asarray(y_prob, dtype=float).reshape(-1)
    if len(y_true) != len(y_prob) or not len(y_true):
        raise ValueError("y_true and y_prob must be equally sized and non-empty")
    if not np.isfinite(y_prob).all() or not 0 <= threshold <= 1:
        raise ValueError("Probabilities must be finite and threshold must be in [0, 1]")
    y_pred_binary = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary, labels=[0, 1]).ravel()
    real_support, fake_support = int(tn + fp), int(tp + fn)
    predicted_real = int(tn + fn)
    sp = None if not real_support else float(tn / real_support)
    se = None if not fake_support else float(tp / fake_support)
    npv = None if not predicted_real else float(tn / predicted_real)
    real_p = precision_score(y_true, y_pred_binary, pos_label=0, zero_division=0)
    real_r = recall_score(y_true, y_pred_binary, pos_label=0, zero_division=0)
    real_f = f1_score(y_true, y_pred_binary, pos_label=0, zero_division=0)
    fake_p = precision_score(y_true, y_pred_binary, pos_label=1, zero_division=0)
    fake_r = recall_score(y_true, y_pred_binary, pos_label=1, zero_division=0)
    fake_f = f1_score(y_true, y_pred_binary, pos_label=1, zero_division=0)
    ranking_available = len(np.unique(y_true)) == 2
    return {
        # Legacy flat keys are intentionally preserved.
        "accuracy": float(accuracy_score(y_true, y_pred_binary)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred_binary)),
        "precision": float(fake_p), "recall": float(fake_r), "f1": float(fake_f),
        "specificity": sp, "sensitivity": se, "mcc": _finite(matthews_corrcoef(y_true, y_pred_binary)),
        "cohen_kappa": _finite(cohen_kappa_score(y_true, y_pred_binary)), "npv": npv,
        "fpr": None if sp is None else float(1 - sp), "fnr": None if se is None else float(1 - se),
        "roc_auc": _finite(roc_auc_score(y_true, y_prob)) if ranking_available else None,
        "pr_auc": _finite(average_precision_score(y_true, y_prob)) if ranking_available else None,
        "true_positives": int(tp), "true_negatives": int(tn), "false_positives": int(fp), "false_negatives": int(fn),
        "threshold": float(threshold),
        "class_metrics": {
            "REAL": {"precision": float(real_p), "recall": float(real_r), "f1": float(real_f), "support": real_support},
            "FAKE": {"precision": float(fake_p), "recall": float(fake_r), "f1": float(fake_f), "support": fake_support},
            "macro_precision": float((real_p + fake_p) / 2), "macro_recall": float((real_r + fake_r) / 2), "macro_f1": float((real_f + fake_f) / 2),
            "weighted_precision": float(precision_score(y_true, y_pred_binary, average="weighted", zero_division=0)),
            "weighted_recall": float(recall_score(y_true, y_pred_binary, average="weighted", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred_binary, average="weighted", zero_division=0)),
        },
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def threshold_analysis(y_true: np.ndarray, y_prob: np.ndarray, step: float = 0.05) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Evaluate 0.05..0.95; call with validation predictions only."""
    if not 0 < step <= 1:
        raise ValueError("step must be in (0, 1]")
    rows = []
    for threshold in np.round(np.arange(0.05, 0.951, step), 10):
        m = compute_all_metrics(y_true, None, y_prob, float(threshold))
        rows.append({key: m[key] for key in ("threshold", "accuracy", "balanced_accuracy", "precision", "recall", "f1", "specificity", "sensitivity", "mcc", "true_positives", "true_negatives", "false_positives", "false_negatives")})
    def best(key: str) -> float:
        valid = [row for row in rows if row[key] is not None]
        return float(max(valid, key=lambda row: (row[key], -row["threshold"]))["threshold"])
    return rows, {"best_f1_threshold": best("f1"), "best_balanced_accuracy_threshold": best("balanced_accuracy"), "best_mcc_threshold": best("mcc")}


def calibration_statistics(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
    """Return Brier score, reliability-bin data, and expected calibration error."""
    y_true, y_prob = _labels(y_true), np.asarray(y_prob, dtype=float).reshape(-1)
    if len(y_true) != len(y_prob) or not len(y_true):
        raise ValueError("Calibration requires equally sized non-empty arrays")
    edges, bin_ids = np.linspace(0, 1, n_bins + 1), None
    bin_ids = np.clip(np.digitize(y_prob, edges[1:-1]), 0, n_bins - 1)
    bins, ece = [], 0.0
    for i in range(n_bins):
        mask, count = bin_ids == i, int((bin_ids == i).sum())
        mean_p = float(y_prob[mask].mean()) if count else None
        observed = float(y_true[mask].mean()) if count else None
        if count: ece += count / len(y_true) * abs(mean_p - observed)
        bins.append({"bin": i, "lower": float(edges[i]), "upper": float(edges[i + 1]), "count": count, "mean_probability": mean_p, "observed_frequency": observed})
    return {"brier_score": float(np.mean((y_prob - y_true) ** 2)), "ece": float(ece), "n_bins": n_bins, "bins": bins}


def compute_roc_curve(
    y_true: np.ndarray, y_prob: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ROC curve data."""
    fpr, tpr, thresholds = roc_curve(_labels(y_true), y_prob)
    return fpr, tpr, thresholds


def compute_pr_curve(
    y_true: np.ndarray, y_prob: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Precision-Recall curve data."""
    precision, recall, thresholds = precision_recall_curve(_labels(y_true), y_prob)
    return precision, recall, thresholds


def format_metrics_table(metrics: Dict[str, float], model_name: str = "") -> str:
    """Format metrics as a readable markdown table."""
    rows = []
    rows.append(f"| Metric | Value |")
    rows.append(f"|--------|-------|")

    display_keys = [
        "accuracy", "precision", "recall", "f1",
        "roc_auc", "pr_auc", "specificity", "sensitivity", "balanced_accuracy", "mcc",
    ]

    for key in display_keys:
        if key in metrics:
            value = metrics[key]
            rows.append(f"| {key.replace('_', ' ').title()} | {'N/A' if value is None else f'{value:.4f}'} |")

    if model_name:
        header = f"\n### {model_name}\n"
        return header + "\n".join(rows)

    return "\n".join(rows)


def format_comparison_table(
    results: Dict[str, Dict[str, float]],
    metrics_to_show: List[str] = None,
) -> str:
    """
    Format a multi-model comparison as a markdown table.

    Args:
        results: Dict mapping model_name → metrics dict.
        metrics_to_show: Which metrics to include.

    Returns:
        Markdown table string.
    """
    if metrics_to_show is None:
        metrics_to_show = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]

    header = "| Model | " + " | ".join(m.replace("_", " ").title() for m in metrics_to_show) + " |"
    sep = "|-------|" + "|".join(["-------"] * len(metrics_to_show)) + "|"
    rows = [header, sep]

    for model_name, metrics in results.items():
        values = ["N/A" if metrics.get(m) is None else f"{metrics.get(m, 0.0):.4f}" for m in metrics_to_show]
        rows.append(f"| {model_name} | " + " | ".join(values) + " |")

    return "\n".join(rows)
