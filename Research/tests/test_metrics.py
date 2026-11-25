"""
DeepShield-X — Metrics Tests

Verify metric calculations for binary classification.
"""

import numpy as np
import pytest

from src.evaluation.metrics import compute_all_metrics, format_metrics_table, format_comparison_table


class TestMetrics:
    """Test metric computation."""

    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.1, 0.9, 0.8, 0.95])
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = compute_all_metrics(y_true, y_pred, y_prob)

        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["true_positives"] == 3
        assert metrics["true_negatives"] == 3
        assert metrics["false_positives"] == 0
        assert metrics["false_negatives"] == 0

    def test_all_wrong_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.9, 0.8, 0.1, 0.2])
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = compute_all_metrics(y_true, y_pred, y_prob)

        assert metrics["accuracy"] == 0.0
        assert metrics["false_positives"] == 2
        assert metrics["false_negatives"] == 2

    def test_mixed_predictions(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.2, 0.6, 0.8, 0.3, 0.1, 0.9])
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = compute_all_metrics(y_true, y_pred, y_prob)

        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1"] <= 1
        assert "roc_auc" in metrics

    def test_format_metrics_table(self):
        metrics = {"accuracy": 0.95, "precision": 0.93, "recall": 0.92, "f1": 0.925}
        table = format_metrics_table(metrics, "TestModel")
        assert "TestModel" in table
        assert "0.9500" in table

    def test_format_comparison_table(self):
        results = {
            "ModelA": {"accuracy": 0.9, "f1": 0.85},
            "ModelB": {"accuracy": 0.95, "f1": 0.92},
        }
        table = format_comparison_table(results, ["accuracy", "f1"])
        assert "ModelA" in table
        assert "ModelB" in table
