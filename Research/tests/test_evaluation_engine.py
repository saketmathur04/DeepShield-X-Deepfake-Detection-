"""Fast, array-only tests for the research-grade evaluation engine."""

import json

import numpy as np
import pytest

from src.evaluation.artifacts import write_evaluation_package
from src.evaluation.metrics import calibration_statistics, compute_all_metrics, threshold_analysis


def test_extended_binary_metrics_are_correct():
    # TN=1, FP=1, FN=1, TP=1 at threshold .5.
    metrics = compute_all_metrics(np.array([0, 0, 1, 1]), None, np.array([.1, .9, .2, .8]))
    assert metrics["accuracy"] == pytest.approx(.5)
    assert metrics["balanced_accuracy"] == pytest.approx(.5)
    assert metrics["specificity"] == pytest.approx(.5)
    assert metrics["sensitivity"] == pytest.approx(.5)
    assert metrics["npv"] == pytest.approx(.5)
    assert metrics["fpr"] == pytest.approx(.5)
    assert metrics["fnr"] == pytest.approx(.5)
    assert metrics["mcc"] == pytest.approx(0)
    assert metrics["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert metrics["class_metrics"]["REAL"]["support"] == 2


def test_zero_division_and_single_class_ranking_are_explicit():
    metrics = compute_all_metrics(np.array([0, 0]), None, np.array([.1, .2]))
    assert metrics["precision"] == 0
    assert metrics["roc_auc"] is None
    assert metrics["pr_auc"] is None


def test_threshold_and_calibration_analysis():
    y_true, y_prob = np.array([0, 0, 1, 1]), np.array([.1, .4, .6, .9])
    rows, selection = threshold_analysis(y_true, y_prob, .05)
    assert len(rows) == 19
    assert set(selection) == {"best_f1_threshold", "best_balanced_accuracy_threshold", "best_mcc_threshold"}
    calibration = calibration_statistics(y_true, y_prob)
    assert calibration["brier_score"] >= 0
    assert len(calibration["bins"]) == 10
    json.dumps(calibration, allow_nan=False)


def test_artifact_package_writes_backward_compatible_serializable_outputs(tmp_path):
    result = write_evaluation_package(
        y_true=np.array([0, 0, 1, 1]), y_prob=np.array([.1, .7, .3, .9]),
        metadata=[{"image_path": "missing.jpg", "source_id": "a"}, {"image_path": "missing.jpg", "source_id": "b"}, {"image_path": "missing.jpg", "source_id": "c"}, {"image_path": "missing.jpg", "source_id": "d"}],
        output_dir=str(tmp_path), experiment_id="unit", model_name="custom_cnn", split="val", threshold=.5,
    )
    assert result["metrics"]["true_negatives"] == 1
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "errors" / "prediction_details.csv").exists()
    assert (tmp_path / "threshold_metrics.json").exists()
    with open(tmp_path / "metrics.json", encoding="utf-8") as handle:
        assert json.load(handle)["positive_class"] == "FAKE"
