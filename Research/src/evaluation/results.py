"""Discovery and aggregation of completed evaluation packages."""

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd


FIELDS = ["Model", "Experiment", "Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1", "Sensitivity", "Specificity", "MCC", "ROC-AUC", "PR-AUC", "TN", "FP", "FN", "TP", "Parameters", "Best Epoch", "Training Time"]


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle: return json.load(handle)


def discover_completed_results(results_dir: str) -> List[Dict[str, Any]]:
    """Discover standardized metrics.json plus legacy *_metrics.json safely."""
    rows, seen = [], set()
    for path in Path(results_dir).rglob("metrics.json"):
        data = _read_json(path)
        if not isinstance(data.get("metrics"), dict): continue
        if data.get("dataset") != "FaceForensics++ C23": continue
        if data.get("split") != "test": continue  # Master comparison is test-only.
        metrics, cm = data["metrics"], data.get("confusion_matrix", {})
        key = (data.get("experiment_id", path.parent.parent.name), data.get("model", path.parent.name))
        seen.add(key); rows.append(_row(key[0], key[1], metrics, cm, path.parent / "experiment_summary.json"))
    for path in Path(results_dir).rglob("*_metrics.json"):
        data = _read_json(path); metrics = data.get("metrics", data)
        if not isinstance(metrics, dict) or "accuracy" not in metrics: continue
        if data.get("timing", {}).get("num_samples", 1000) < 1000: continue
        experiment, model = path.parent.name, data.get("model_name", path.stem.replace("_metrics", ""))
        if (experiment, model) not in seen: rows.append(_row(experiment, model, metrics, metrics.get("confusion_matrix", {}), None))
    return rows


def _row(experiment: str, model: str, metrics: Dict[str, Any], cm: Dict[str, Any], summary_path: Path | None) -> Dict[str, Any]:
    summary = _read_json(summary_path) if summary_path and summary_path.exists() else {}
    return {"Model": model, "Experiment": experiment, "Accuracy": metrics.get("accuracy"), "Balanced Accuracy": metrics.get("balanced_accuracy"), "Precision": metrics.get("precision"), "Recall": metrics.get("recall"), "F1": metrics.get("f1"), "Sensitivity": metrics.get("sensitivity"), "Specificity": metrics.get("specificity"), "MCC": metrics.get("mcc"), "ROC-AUC": metrics.get("roc_auc"), "PR-AUC": metrics.get("pr_auc"), "TN": cm.get("tn", metrics.get("true_negatives")), "FP": cm.get("fp", metrics.get("false_positives")), "FN": cm.get("fn", metrics.get("false_negatives")), "TP": cm.get("tp", metrics.get("true_positives")), "Parameters": summary.get("parameter_count"), "Best Epoch": summary.get("best_epoch"), "Training Time": summary.get("training_time_seconds")}


def update_master_results(results_dir: str) -> pd.DataFrame:
    root = Path(results_dir); root.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(discover_completed_results(str(root)), columns=FIELDS)
    frame.to_csv(root / "master_results.csv", index=False)
    with open(root / "master_results.json", "w", encoding="utf-8") as handle: json.dump(frame.where(pd.notna(frame), None).to_dict(orient="records"), handle, indent=2)
    if len(frame) >= 2:
        for column, slug in (("Accuracy", "accuracy"), ("Balanced Accuracy", "balanced_accuracy"), ("F1", "f1"), ("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc"), ("Specificity", "specificity")):
            chart = frame.dropna(subset=[column])
            if len(chart) < 2: continue
            fig, ax = plt.subplots(figsize=(8, 5)); ax.bar(chart["Experiment"], chart[column]); ax.set(ylim=(0, 1.05), ylabel=column, title=f"Model Comparison — {column}"); ax.tick_params(axis="x", rotation=25); fig.tight_layout(); fig.savefig(root / f"model_comparison_{slug}.png", dpi=300); plt.close(fig)
    return frame
