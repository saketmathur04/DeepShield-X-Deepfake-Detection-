"""
    
DeepShield-X — Error Analysis

Identifies false positives, false negatives, low-confidence predictions,
and high-confidence incorrect predictions.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.utils.visualization import plot_confidence_distribution

logger = logging.getLogger(__name__)


def analyze_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    metadata: List[Dict],
    output_dir: Optional[str] = None,
    top_k: int = 10,
) -> Dict[str, Any]:
    """
    Comprehensive error analysis.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities.
        metadata: List of metadata dicts per sample.
        output_dir: Directory to save analysis.
        top_k: Number of top examples to include.

    Returns:
        Dict with error analysis results.
    """
    is_correct = y_true == y_pred
    confidence = np.where(y_pred == 1, y_prob, 1 - y_prob)

    # Identify error types
    false_positives = np.where((y_true == 0) & (y_pred == 1))[0]
    false_negatives = np.where((y_true == 1) & (y_pred == 0))[0]
    correct_mask = y_true == y_pred

    # High confidence errors
    incorrect_mask = ~correct_mask
    incorrect_conf = confidence[incorrect_mask]
    incorrect_indices = np.where(incorrect_mask)[0]

    if len(incorrect_conf) > 0:
        high_conf_incorrect_idx = incorrect_indices[
            np.argsort(-incorrect_conf)[:top_k]
        ]
    else:
        high_conf_incorrect_idx = []

    # Low confidence correct predictions
    correct_conf = confidence[correct_mask]
    correct_indices = np.where(correct_mask)[0]

    if len(correct_conf) > 0:
        low_conf_correct_idx = correct_indices[
            np.argsort(correct_conf)[:top_k]
        ]
    else:
        low_conf_correct_idx = []

    # Build result
    def _sample_info(indices, limit=top_k):
        """Extract sample info for given indices."""
        results = []
        for idx in indices[:limit]:
            info = {
                "index": int(idx),
                "true_label": "FAKE" if y_true[idx] == 1 else "REAL",
                "predicted_label": "FAKE" if y_pred[idx] == 1 else "REAL",
                "probability": round(float(y_prob[idx]), 4),
                "confidence": round(float(confidence[idx]), 4),
            }
            if idx < len(metadata):
                info["image_path"] = metadata[idx].get("image_path", "unknown")
                info["source_id"] = metadata[idx].get("source_id", "unknown")
            results.append(info)
        return results

    analysis = {
        "total_samples": len(y_true),
        "total_correct": int(correct_mask.sum()),
        "total_incorrect": int(incorrect_mask.sum()),
        "accuracy": float(correct_mask.mean()),
        "false_positives": {
            "count": len(false_positives),
            "description": "REAL images incorrectly predicted as FAKE",
            "examples": _sample_info(false_positives),
        },
        "false_negatives": {
            "count": len(false_negatives),
            "description": "FAKE images incorrectly predicted as REAL",
            "examples": _sample_info(false_negatives),
        },
        "high_confidence_errors": {
            "count": len(high_conf_incorrect_idx),
            "description": "Incorrect predictions with highest confidence",
            "examples": _sample_info(high_conf_incorrect_idx),
        },
        "low_confidence_correct": {
            "count": len(low_conf_correct_idx),
            "description": "Correct predictions with lowest confidence",
            "examples": _sample_info(low_conf_correct_idx),
        },
        "confidence_stats": {
            "mean_correct": float(correct_conf.mean()) if len(correct_conf) > 0 else 0,
            "mean_incorrect": float(incorrect_conf.mean()) if len(incorrect_conf) > 0 else 0,
            "median_correct": float(np.median(correct_conf)) if len(correct_conf) > 0 else 0,
            "median_incorrect": float(np.median(incorrect_conf)) if len(incorrect_conf) > 0 else 0,
        },
    }

    # Save
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        with open(out_path / "error_analysis.json", "w") as f:
            json.dump(analysis, f, indent=2)

        # Confidence distribution plot
        plot_confidence_distribution(
            y_true, y_prob,
            save_path=str(out_path / "confidence_distribution.png"),
        )

        # Markdown report
        _write_error_report(analysis, out_path / "error_analysis.md")

        logger.info(f"Error analysis saved to: {out_path}")

    return analysis


def _write_error_report(analysis: Dict, path: Path) -> None:
    """Write error analysis as markdown."""
    lines = [
        "# Error Analysis Report\n",
        f"**Total Samples**: {analysis['total_samples']}",
        f"**Accuracy**: {analysis['accuracy']:.4f}",
        f"**Correct**: {analysis['total_correct']} | **Incorrect**: {analysis['total_incorrect']}\n",
        "## False Positives (REAL → FAKE)",
        f"Count: {analysis['false_positives']['count']}\n",
    ]

    for ex in analysis["false_positives"]["examples"][:5]:
        lines.append(f"- {ex.get('image_path', 'N/A')} (conf: {ex['confidence']:.4f})")

    lines.extend([
        "\n## False Negatives (FAKE → REAL)",
        f"Count: {analysis['false_negatives']['count']}\n",
    ])

    for ex in analysis["false_negatives"]["examples"][:5]:
        lines.append(f"- {ex.get('image_path', 'N/A')} (conf: {ex['confidence']:.4f})")

    lines.extend([
        "\n## Confidence Analysis",
        f"- Mean confidence (correct): {analysis['confidence_stats']['mean_correct']:.4f}",
        f"- Mean confidence (incorrect): {analysis['confidence_stats']['mean_incorrect']:.4f}",
    ])

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
