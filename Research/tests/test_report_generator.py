"""Regression test: nested evaluation artifacts must not become experiments."""

import json

from scripts.generate_report import build_report, discover_experiments


def test_report_discovers_only_registered_experiment_metrics(tmp_path):
    experiment = tmp_path / "efficientnet" / "evaluation" / "test"
    (experiment / "curves").mkdir(parents=True)
    metrics = {"metrics": {"accuracy": .9, "precision": .9, "recall": .9, "specificity": .9, "f1": .9, "roc_auc": .9, "pr_auc": .9, "balanced_accuracy": .9, "mcc": .8, "threshold": .5}, "confusion_matrix": {"tn": 1, "fp": 0, "fn": 0, "tp": 1}}
    (experiment / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (experiment / "curves" / "calibration.json").write_text(json.dumps({"ece": .1}), encoding="utf-8")

    records = discover_experiments(tmp_path)
    assert len(records) == 8
    assert next(record for record in records if record["id"] == "efficientnet")["available"]
    report = build_report(tmp_path)
    assert "curves — calibration" not in report
    assert "## 6. Primary Architecture Comparison" in report
