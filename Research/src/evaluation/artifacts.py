"""Persist reproducible, publication-ready evaluation artifacts.

This module is intentionally separate from model inference.  It accepts CPU
arrays and metadata so it can be used by every architecture without changing
the dataset split or retaining tensors on the GPU.
"""

import csv
import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from src.evaluation.metrics import calibration_statistics, compute_all_metrics, threshold_analysis
from src.utils.seed import get_environment_info

PLOT_DPI = 300
LABEL_NAMES = {0: "REAL", 1: "FAKE"}


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, allow_nan=False, default=str)


def _save_figure(fig: plt.Figure, path: Path, pdf: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    if pdf:
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def prediction_frame(y_true: np.ndarray, y_prob: np.ndarray, metadata: List[Dict[str, Any]], threshold: float) -> pd.DataFrame:
    """Create the canonical per-frame prediction table without duplicate fields."""
    y_true, y_prob = np.asarray(y_true, dtype=int), np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    rows = []
    for index, (actual, predicted, probability) in enumerate(zip(y_true, y_pred, y_prob)):
        meta = metadata[index] if index < len(metadata) else {}
        error_type = ("TP" if actual else "TN") if actual == predicted else ("FP" if predicted else "FN")
        rows.append({
            "sample_path": meta.get("image_path", ""), "source_id": meta.get("source_id", ""),
            "true_label": int(actual), "true_label_name": LABEL_NAMES[int(actual)],
            "predicted_label": int(predicted), "predicted_label_name": LABEL_NAMES[int(predicted)],
            "probability": float(probability), "confidence": float(probability if predicted else 1 - probability),
            "threshold": float(threshold), "correct": bool(actual == predicted), "error_type": error_type,
        })
    return pd.DataFrame(rows)


def confidence_statistics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Dict[str, Optional[float]]]:
    result = {}
    for label, name in LABEL_NAMES.items():
        values = np.asarray(y_prob, dtype=float)[np.asarray(y_true, dtype=int) == label]
        result[name] = {key: (float(value) if len(values) else None) for key, value in {
            "mean": values.mean() if len(values) else None, "median": np.median(values) if len(values) else None,
            "std": values.std() if len(values) else None, "min": values.min() if len(values) else None,
            "max": values.max() if len(values) else None, "p25": np.percentile(values, 25) if len(values) else None,
            "p75": np.percentile(values, 75) if len(values) else None}.items()}
    return result


def dataset_statistics(splits_dir: Path) -> Optional[Dict[str, Any]]:
    """Read existing split CSVs only; this function never changes them."""
    splits = {}
    for split in ("train", "val", "test"):
        csv_path = splits_dir / f"{split}.csv"
        if not csv_path.exists():
            return None
        frame = pd.read_csv(csv_path, usecols=["label"])
        real, fake = int((frame.label == 0).sum()), int((frame.label == 1).sum())
        total = len(frame)
        splits[split] = {"REAL": real, "FAKE": fake, "total": total,
                         "real_prevalence": real / total if total else None,
                         "fake_prevalence": fake / total if total else None,
                         "fake_to_real_ratio": fake / real if real else None}
    return splits


def _plot_curves(y_true: np.ndarray, y_prob: np.ndarray, out: Path, model_name: str) -> None:
    curves = out / "curves"
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thresholds}).to_csv(curves / "roc_curve.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 6)); ax.plot(fpr, tpr, label=f"{model_name} (ROC-AUC={roc_auc_score(y_true, y_prob):.4f})"); ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve", xlim=(0, 1), ylim=(0, 1)); ax.legend(); ax.grid(alpha=.3)
    _save_figure(fig, curves / "roc_curve.png", pdf=True)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
    pd.DataFrame({"precision": precision[:-1], "recall": recall[:-1], "threshold": pr_thresholds}).to_csv(curves / "precision_recall_curve.csv", index=False)
    prevalence = float(np.mean(y_true))
    fig, ax = plt.subplots(figsize=(7, 6)); ax.plot(recall, precision, label=f"{model_name} (AP={average_precision_score(y_true, y_prob):.4f})"); ax.axhline(prevalence, color="k", ls="--", label=f"FAKE prevalence ({prevalence:.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve", xlim=(0, 1), ylim=(0, 1)); ax.legend(); ax.grid(alpha=.3)
    _save_figure(fig, curves / "precision_recall_curve.png", pdf=True)


def _plot_confusion(metrics: Dict[str, Any], out: Path) -> None:
    cm = np.array([[metrics["true_negatives"], metrics["false_positives"]], [metrics["false_negatives"], metrics["true_positives"]]])
    for normalized, name, fmt in ((False, "confusion_matrix_counts.png", "d"), (True, "confusion_matrix_normalized.png", ".1%")):
        values = cm / cm.sum(axis=1, keepdims=True) if normalized else cm
        fig, ax = plt.subplots(figsize=(6, 5)); sns.heatmap(values, annot=True, fmt=fmt, cmap="Blues", xticklabels=["REAL", "FAKE"], yticklabels=["REAL", "FAKE"], ax=ax)
        ax.set(xlabel="Predicted", ylabel="Actual", title="Confusion Matrix" + (" (row-normalized)" if normalized else " (counts)")); _save_figure(fig, out / "curves" / name)


def _plot_confidence(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, name, color in ((0, "REAL", "#377eb8"), (1, "FAKE", "#e41a1c")):
        ax.hist(np.asarray(y_prob)[np.asarray(y_true) == label], bins=30, alpha=.55, label=name, color=color)
    ax.axvline(threshold, color="black", ls="--", label=f"Threshold {threshold:.2f}"); ax.set(xlabel="Predicted FAKE probability", ylabel="Frame count", title="Confidence Distribution"); ax.legend()
    _save_figure(fig, out / "curves" / "confidence_distribution.png")


def _plot_calibration(calibration: Dict[str, Any], out: Path) -> None:
    valid = [b for b in calibration["bins"] if b["count"]]
    fig, ax = plt.subplots(figsize=(6, 6)); ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    if valid: ax.plot([b["mean_probability"] for b in valid], [b["observed_frequency"] for b in valid], "o-", label="Model")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed FAKE frequency", title=f"Calibration Curve (ECE={calibration['ece']:.4f})", xlim=(0, 1), ylim=(0, 1)); ax.legend(); ax.grid(alpha=.3)
    _save_figure(fig, out / "curves" / "calibration_curve.png")


def _error_gallery(frame: pd.DataFrame, selector: str, out_path: Path, limit: int = 16) -> bool:
    selected = frame[frame.error_type == selector].sort_values("confidence", ascending=False).head(limit)
    if selected.empty: return False
    fig, axes = plt.subplots(4, 4, figsize=(12, 12)); axes = axes.ravel()
    for ax, (_, row) in zip(axes, selected.iterrows()):
        try:
            with Image.open(row.sample_path) as image: ax.imshow(image.convert("RGB"))
            ax.set_title(f"{row.true_label_name} → {row.predicted_label_name}\np={row.probability:.3f}, c={row.confidence:.3f}", fontsize=8)
        except (FileNotFoundError, OSError, ValueError):
            ax.text(.5, .5, "Image unavailable", ha="center", va="center")
        ax.axis("off")
    for ax in axes[len(selected):]: ax.axis("off")
    _save_figure(fig, out_path); return True


def _video_level(frame: pd.DataFrame, out: Path, threshold: float) -> Dict[str, Any]:
    usable = frame.source_id.fillna("").astype(str)
    if frame.empty or (usable == "").any() or (usable == "unknown").any():
        return {"video_level_evaluation_available": False, "reason": "Reliable source_id is unavailable for one or more frames."}
    grouped = frame.assign(source_id=usable).groupby("source_id", as_index=False).agg(true_label=("true_label", "first"), frames=("true_label", "size"), mean_probability=("probability", "mean"), median_probability=("probability", "median"), max_probability=("probability", "max"))
    # A source must have a single label; otherwise video aggregation is unsafe.
    if frame.groupby("source_id").true_label.nunique().max() != 1:
        return {"video_level_evaluation_available": False, "reason": "A source_id contains mixed labels."}
    grouped["video_prediction"] = (grouped.mean_probability >= threshold).astype(int)
    grouped.to_csv(out / "video" / "video_predictions.csv", index=False)
    metrics = compute_all_metrics(grouped.true_label.to_numpy(), grouped.video_prediction.to_numpy(), grouped.mean_probability.to_numpy(), threshold)
    result = {"video_level_evaluation_available": True, "num_videos": len(grouped), "metrics": metrics}
    _write_json(out / "video" / "video_metrics.json", result); return result


def collect_environment() -> Dict[str, Any]:
    result = get_environment_info(); result.update({"hostname": socket.gethostname(), "timestamp": datetime.now(timezone.utc).isoformat(), "platform": platform.platform()})
    try: result["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError): result["git_commit"] = None
    try:
        import cv2; result["opencv_version"] = cv2.__version__  # noqa: E702
    except ImportError: result["opencv_version"] = None
    try:
        import pandas; result["pandas_version"] = pandas.__version__
        import sklearn; result["scikit_learn_version"] = sklearn.__version__
    except ImportError: pass
    return result


def write_training_history_plots(history: Dict[str, List[Any]], output_dir: str, model_name: str) -> List[str]:
    """Write only curves that actually exist; never invent missing old history."""
    out = Path(output_dir) / "training"; out.mkdir(parents=True, exist_ok=True)
    specs = (("train_loss", "training_loss.png", "Training Loss"), ("val_loss", "validation_loss.png", "Validation Loss"),
             ("train_acc", "training_accuracy.png", "Training Accuracy"), ("val_acc", "validation_accuracy.png", "Validation Accuracy"),
             ("learning_rate", "learning_rate.png", "Learning Rate"))
    written = []
    for key, filename, title in specs:
        values = history.get(key, [])
        if not values: continue
        fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(range(1, len(values) + 1), values, marker="o")
        ax.set(xlabel="Epoch", ylabel=title, title=f"{model_name} — {title}"); ax.grid(alpha=.3)
        path = out / filename; _save_figure(fig, path); written.append(str(path))
    for key in ("val_precision", "val_recall", "val_f1", "val_specificity", "val_balanced_accuracy", "val_roc_auc", "val_pr_auc"):
        values = history.get(key, [])
        if not values or all(value is None for value in values): continue
        fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(range(1, len(values) + 1), values, marker="o")
        ax.set(xlabel="Epoch", ylabel=key.replace("val_", "").replace("_", " ").title(), title=f"{model_name} — {key.replace('val_', '').replace('_', ' ').title()}", ylim=(0, 1.05)); ax.grid(alpha=.3)
        path = out / f"{key}.png"; _save_figure(fig, path); written.append(str(path))
    return written


def write_evaluation_package(*, y_true: np.ndarray, y_prob: np.ndarray, metadata: List[Dict[str, Any]], output_dir: str, experiment_id: str, model_name: str, split: str, threshold: float, config: Optional[Dict[str, Any]] = None, checkpoint_path: Optional[str] = None, threshold_step: float = .05, splits_dir: Optional[Path] = None, dataset_name: str = "FaceForensics++ C23") -> Dict[str, Any]:
    """Write a complete package to a new/additive evaluation directory."""
    out = Path(output_dir); (out / "curves").mkdir(parents=True, exist_ok=True); (out / "errors").mkdir(exist_ok=True); (out / "video").mkdir(exist_ok=True)
    metrics = compute_all_metrics(y_true, None, y_prob, threshold)
    frame = prediction_frame(y_true, y_prob, metadata, threshold)
    frame.to_csv(out / "errors" / "prediction_details.csv", index=False)
    for kind, file_name in (("FP", "false_positives.csv"), ("FN", "false_negatives.csv")):
        frame[frame.error_type == kind].sort_values("confidence", ascending=False).to_csv(out / "errors" / file_name, index=False)
    frame[~frame.correct].sort_values("confidence", ascending=False).to_csv(out / "errors" / "misclassified_samples.csv", index=False)
    frame[(~frame.correct) & (frame.confidence >= .90)].sort_values("confidence", ascending=False).to_csv(out / "errors" / "high_confidence_errors.csv", index=False)
    _write_json(out / "metrics.json", {"experiment_id": experiment_id, "model": model_name, "dataset": dataset_name, "split": split, "positive_class": "FAKE", "negative_class": "REAL", "metrics": {k: v for k, v in metrics.items() if k not in ("class_metrics", "confusion_matrix")}, "class_metrics": {"REAL": metrics["class_metrics"]["REAL"], "FAKE": metrics["class_metrics"]["FAKE"], "macro_precision": metrics["class_metrics"]["macro_precision"], "macro_recall": metrics["class_metrics"]["macro_recall"], "macro_f1": metrics["class_metrics"]["macro_f1"], "weighted_precision": metrics["class_metrics"]["weighted_precision"], "weighted_recall": metrics["class_metrics"]["weighted_recall"], "weighted_f1": metrics["class_metrics"]["weighted_f1"]}, "confusion_matrix": metrics["confusion_matrix"], "threshold": threshold})
    pd.DataFrame([metrics]).drop(columns=["class_metrics", "confusion_matrix"], errors="ignore").to_csv(out / "metrics.csv", index=False)
    _plot_curves(y_true, y_prob, out, model_name); _plot_confusion(metrics, out); _plot_confidence(y_true, y_prob, threshold, out)
    confidence = confidence_statistics(y_true, y_prob); _write_json(out / "curves" / "confidence_statistics.json", confidence)
    calibration = calibration_statistics(y_true, y_prob); pd.DataFrame(calibration["bins"]).to_csv(out / "curves" / "calibration.csv", index=False); _write_json(out / "curves" / "calibration.json", calibration); _plot_calibration(calibration, out)
    if split == "val":
        rows, selection = threshold_analysis(y_true, y_prob, threshold_step); pd.DataFrame(rows).to_csv(out / "threshold_metrics.csv", index=False); _write_json(out / "threshold_metrics.json", {"split": "val", "selection": selection, "rows": rows})
    else:
        _write_json(out / "threshold_metrics.json", {"not_run": True, "reason": "Threshold selection is validation-only; test threshold was not tuned."})
    for kind, name in (("FP", "error_gallery_fp.png"), ("FN", "error_gallery_fn.png"), ("TN", "correct_gallery_real.png"), ("TP", "correct_gallery_fake.png")):
        _error_gallery(frame, kind, out / "errors" / name)
    video = _video_level(frame, out, threshold)
    environment = collect_environment(); _write_json(out / "environment.json", environment)
    configuration = config or {}; training = configuration.get("training", {}); model_cfg = configuration.get("model", {})
    summary = {"experiment_id": experiment_id, "model": model_name, "dataset": dataset_name, "split": split, "checkpoint_path": checkpoint_path, "evaluation_timestamp": environment["timestamp"], "random_seed": configuration.get("project", {}).get("seed"), "epochs": training.get("epochs"), "batch_size": training.get("batch_size"), "learning_rate": training.get("learning_rate"), "optimizer": training.get("optimizer"), "scheduler": training.get("scheduler"), "pos_weight": training.get("pos_weight"), "augmentation": configuration.get("augmentation"), "pretrained": model_cfg.get("pretrained"), "video_level": video}
    if splits_dir:
        summary["dataset_statistics"] = dataset_statistics(splits_dir)
    _write_json(out / "experiment_summary.json", summary)
    return {"metrics": metrics, "output_dir": str(out), "video": video}
