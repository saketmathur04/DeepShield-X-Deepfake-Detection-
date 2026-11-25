"""
DeepShield-X — Data Loader Service

Dynamically reads all experimental JSON result files.
Caches based on file modification time so it never serves stale data
but doesn't re-read on every request.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Resolve project root (DEEPSHIELD-X/) relative to this file
_BACKEND_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BACKEND_DIR.parent
_PROJECT_ROOT = _REPO_ROOT / "Research"
_EXPERIMENTS_DIR = _PROJECT_ROOT / "experiments"
_RESULTS_DIR = _EXPERIMENTS_DIR / "results"
_DOCS_IMAGES = _PROJECT_ROOT / "docs" / "images"


class _CachedFile:
    """Read a JSON file and cache it until the file changes on disk."""

    def __init__(self, path: Path):
        self._path = path
        self._mtime: float = 0.0
        self._data: Any = None

    def read(self) -> Any:
        if not self._path.exists():
            logger.warning("File not found: %s", self._path)
            return None
        mtime = self._path.stat().st_mtime
        if mtime != self._mtime or self._data is None:
            with open(self._path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
            self._mtime = mtime
            logger.debug("Loaded %s (%.0f bytes)", self._path.name, self._path.stat().st_size)
        return self._data


# Singleton caches for each result file
_experiment_registry = _CachedFile(_EXPERIMENTS_DIR / "experiment_registry.json")
_robustness_results = _CachedFile(_RESULTS_DIR / "robustness" / "robustness_results.json")
_manipulation_results = _CachedFile(_RESULTS_DIR / "manipulation_analysis" / "manipulation_results.json")
_efficiency_results = _CachedFile(_RESULTS_DIR / "efficiency" / "efficiency_results.json")
_master_comparison = _CachedFile(_RESULTS_DIR / "master_benchmark" / "master_comparison.json")
_bootstrap_results = _CachedFile(_RESULTS_DIR / "statistical_analysis" / "bootstrap_results.json")


def get_project_root() -> Path:
    return _PROJECT_ROOT


def get_experiments_dir() -> Path:
    return _EXPERIMENTS_DIR


def get_results_dir() -> Path:
    return _RESULTS_DIR


def get_docs_images_dir() -> Path:
    return _DOCS_IMAGES


# ---------------------------------------------------------------------------
# Experiment Registry
# ---------------------------------------------------------------------------

def load_experiment_registry() -> List[Dict]:
    """Load the master experiment registry."""
    data = _experiment_registry.read()
    return data if data else []


def load_experiment_detail(experiment_id: str) -> Optional[Dict]:
    """
    Load detailed results for a single experiment.
    Merges registry entry with per-experiment metrics.json + config.
    """
    registry = load_experiment_registry()
    entry = None
    for exp in registry:
        if exp.get("experiment_id") == experiment_id:
            entry = dict(exp)  # copy
            break
    if entry is None:
        return None

    # Try to load the detailed metrics.json
    metrics_path = _RESULTS_DIR / experiment_id / "evaluation" / "test" / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            detail = json.load(f)
        entry["detailed_metrics"] = detail.get("metrics", {})
        entry["class_metrics"] = detail.get("class_metrics", {})
        entry["confusion_matrix"] = detail.get("confusion_matrix", {})
        entry["threshold"] = detail.get("threshold", 0.5)
    else:
        # Use the test_metrics from registry as fallback
        tm = entry.get("test_metrics", {})
        entry["detailed_metrics"] = tm
        entry["confusion_matrix"] = None

    # Try to load training history
    history_path = _EXPERIMENTS_DIR / "checkpoints" / experiment_id
    history_files = list(history_path.glob("*_history.json")) if history_path.exists() else []
    if history_files:
        with open(history_files[0], "r", encoding="utf-8") as f:
            entry["training_history"] = json.load(f)
    else:
        entry["training_history"] = None

    # Try to load training config from the checkpoint
    config_path = _RESULTS_DIR / experiment_id / "evaluation" / "test" / "experiment_summary.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        entry["config"] = summary.get("config", summary.get("training_config", {}))
    else:
        entry["config"] = {}

    return entry


# ---------------------------------------------------------------------------
# Analysis Results
# ---------------------------------------------------------------------------

def load_robustness_results() -> List[Dict]:
    data = _robustness_results.read()
    return data if data else []


def load_manipulation_results() -> List[Dict]:
    data = _manipulation_results.read()
    return data if data else []


def load_efficiency_results() -> List[Dict]:
    data = _efficiency_results.read()
    return data if data else []


# ---------------------------------------------------------------------------
# Model Info (merged from registry + efficiency)
# ---------------------------------------------------------------------------

_MODEL_DISPLAY_NAMES = {
    "custom_cnn": "Custom CNN",
    "class_balanced_custom_cnn": "Custom CNN (Balanced)",
    "resnet50": "ResNet50",
    "resnet50_regularized": "ResNet50 Regularized",
    "efficientnet": "EfficientNet-B0",
    "mobilenet": "MobileNetV3-Large",
    "xception": "Xception",
    "vit": "ViT-B/16",
}

_MODEL_FAMILIES = {
    "custom_cnn": "CNN",
    "class_balanced_custom_cnn": "CNN",
    "resnet50": "CNN",
    "resnet50_regularized": "CNN",
    "efficientnet": "CNN",
    "mobilenet": "CNN",
    "xception": "CNN",
    "vit": "Transformer",
}


def load_master_comparison() -> List[Dict]:
    """Load master comparison benchmark metrics for all models."""
    data = _master_comparison.read()
    return data if data else []


def load_bootstrap_results() -> List[Dict]:
    """Load bootstrap statistical confidence interval results."""
    data = _bootstrap_results.read()
    return data if data else []


def load_research_plots() -> List[Dict]:
    """Return available high-res scientific figures generated during research."""
    plot_definitions = [
        {"id": "model_comparison_bar", "title": "Model Comparison Benchmark", "filename": "model_comparison_bar.png", "category": "Benchmark", "description": "Master evaluation of Accuracy, F1-Score, and ROC-AUC across all 8 architectures."},
        {"id": "efficiency_comparison", "title": "Inference Efficiency Tradeoff", "filename": "efficiency_comparison.png", "category": "Efficiency", "description": "Parameter size vs CPU inference latency vs accuracy Pareto frontier."},
        {"id": "robustness_curve", "title": "Perturbation Robustness Curve", "filename": "robustness_curve.png", "category": "Robustness", "description": "Detection retention under JPEG compression, blur, noise, and resolution scale degradation."},
        {"id": "gradcam_sample_grid", "title": "Grad-CAM Attribution Matrix", "filename": "gradcam_sample_grid.png", "category": "Explainability", "description": "Visual saliency heatmaps highlighting facial boundary anomalies and blending artifacts."},
        {"id": "confusion_matrix", "title": "Normalized Confusion Matrix", "filename": "confusion_matrix.png", "category": "Error Analysis", "description": "Per-class classification distribution on frozen held-out test partition (N=9,589)."},
        {"id": "calibration_curve", "title": "Model Reliability Calibration", "filename": "calibration_curve.png", "category": "Error Analysis", "description": "Expected vs observed deepfake probability calibration and reliability diagram."},
        {"id": "confidence_distribution", "title": "Confidence Distribution", "filename": "confidence_distribution.png", "category": "Error Analysis", "description": "Prediction confidence density across real and manipulated samples."},
        {"id": "manipulation_breakdown", "title": "Manipulation Technique Breakdown", "filename": "manipulation_breakdown.png", "category": "Forensics", "description": "Detection performance isolated across DeepFakes, Face2Face, FaceSwap, NeuralTextures."},
        {"id": "training_dynamics", "title": "Loss & Generalization Dynamics", "filename": "training_dynamics.png", "category": "Training", "description": "Convergence trajectories, validation loss progression, and regularization ablation."},
        {"id": "architecture_pipeline", "title": "DeepShield-X System Pipeline", "filename": "architecture_pipeline.png", "category": "Architecture", "description": "End-to-end data pipeline, identity-isolated split, neural feature extraction and explainability."}
    ]
    results = []
    for p in plot_definitions:
        file_path = _DOCS_IMAGES / p["filename"]
        if file_path.exists():
            p_copy = dict(p)
            p_copy["url"] = f"/static/images/{p['filename']}"
            p_copy["size_bytes"] = file_path.stat().st_size
            results.append(p_copy)
    return results


def load_system_info() -> Dict[str, Any]:
    """Return live system telemetry, device info, and inference readiness."""
    import torch
    has_cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if has_cuda else "Host CPU Architecture"
    models = load_models()
    return {
        "status": "ready",
        "framework": f"PyTorch {torch.__version__}",
        "device": "cuda" if has_cuda else "cpu",
        "device_name": device_name,
        "cuda_available": has_cuda,
        "models_count": len(models),
        "models_available": [m["id"] for m in models],
        "benchmark_dataset": "FaceForensics++ (C23)",
        "test_samples": 9589,
        "test_protocol": "Identity-Isolated · Zero-Leakage (S=42)",
    }


def load_efficiency_results() -> List[Dict]:
    """Load hardware performance and computational efficiency benchmarks."""
    data = _efficiency_results.read()
    return data if data else []


def load_dataset_info() -> Dict[str, Any]:
    """Load high-level FaceForensics++ benchmark summary statistics."""
    return {
        "dataset_name": "FaceForensics++ (c23)",
        "total_images": 63940,
        "train_samples": 44758,
        "val_samples": 9591,
        "test_samples": 9591,
        "manipulation_types": ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"],
        "hardware_profile": "NVIDIA A100-SXM4-80GB (CUDA 12.2)",
        "framework": "PyTorch 2.4.0+cu121 · Albumentations",
        "test_protocol": "Identity-Isolated · Zero-Leakage (S=42)",
    }


def load_models() -> List[Dict]:
    """
    Build a unified model list by merging experiment registry with efficiency data.
    Sorted by ROC-AUC descending so the benchmark champion is first.
    """
    registry = load_experiment_registry()
    efficiency = {e["model_id"]: e for e in load_efficiency_results()}

    models = []
    seen_archs = set()
    for exp in registry:
        eid = exp["experiment_id"]
        arch = exp.get("architecture", eid)
        eff = efficiency.get(arch, {})

        checkpoint_dir = _EXPERIMENTS_DIR / "checkpoints" / eid
        if not checkpoint_dir.exists() and eid == "custom_cnn":
            checkpoint_dir = _EXPERIMENTS_DIR / "checkpoints" / "default"
        # A model is selectable for inference only when its own trained
        # checkpoint exists. Registry-only rows remain visible in experiments.
        has_checkpoint = checkpoint_dir.exists() and any(checkpoint_dir.glob("*.pth"))
        model = {
            "id": eid,
            "name": _MODEL_DISPLAY_NAMES.get(eid, eid),
            "architecture": arch,
            "family": _MODEL_FAMILIES.get(arch, "CNN"),
            "pretrained": exp.get("pretrained", False),
            "parameters": _format_params(eff.get("total_parameters")),
            "total_parameters": eff.get("total_parameters"),
            "checkpoint_size_mb": eff.get("checkpoint_size_mb"),
            "cpu_latency_ms": eff.get("cpu_latency_ms"),
            "latency_std_ms": eff.get("latency_std_ms"),
            "throughput_fps": eff.get("throughput_fps"),
            "metrics": exp.get("test_metrics", {}),
            "inference_available": has_checkpoint,
        }
        if has_checkpoint:
            models.append(model)
        seen_archs.add(arch)

    # Sort models by test ROC-AUC descending so the best trained model appears first
    models.sort(
        key=lambda m: float(m.get("metrics", {}).get("roc_auc") or 0.0),
        reverse=True,
    )

    return models


def _format_params(n: Optional[int]) -> str:
    if n is None:
        return "—"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


# ---------------------------------------------------------------------------
# Error Analysis
# ---------------------------------------------------------------------------

def load_error_analysis(experiment_id: str) -> Optional[Dict]:
    """Load error analysis data (FP/FN counts, confidence distribution)."""
    metrics_path = _RESULTS_DIR / experiment_id / "evaluation" / "test" / "metrics.json"
    if not metrics_path.exists():
        return None

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cm = data.get("confusion_matrix", {})
    metrics = data.get("metrics", {})

    return {
        "experiment_id": experiment_id,
        "total_samples": cm.get("tp", 0) + cm.get("tn", 0) + cm.get("fp", 0) + cm.get("fn", 0),
        "total_errors": cm.get("fp", 0) + cm.get("fn", 0),
        "false_positives": cm.get("fp", 0),
        "false_negatives": cm.get("fn", 0),
        "true_positives": cm.get("tp", 0),
        "true_negatives": cm.get("tn", 0),
        "error_rate": 1.0 - metrics.get("accuracy", 0),
        "confusion_matrix": cm,
    }


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def load_reports() -> List[Dict]:
    """Dynamically build report list from available result directories."""
    reports = []
    registry = load_experiment_registry()

    for exp in registry:
        eid = exp["experiment_id"]
        metrics_path = _RESULTS_DIR / eid / "evaluation" / "test" / "metrics.json"
        has_results = metrics_path.exists()
        reports.append({
            "id": eid,
            "name": f"{_MODEL_DISPLAY_NAMES.get(eid, eid)} — Evaluation Report",
            "experiment": eid,
            "status": "Ready for review" if has_results else "Awaiting evaluation",
            "has_results": has_results,
        })

    # Add special analysis reports
    if (_RESULTS_DIR / "robustness" / "robustness_results.json").exists():
        reports.append({
            "id": "robustness_analysis",
            "name": "Robustness Stress-Test Report",
            "experiment": "resnet50_regularized",
            "status": "Ready for review",
            "has_results": True,
        })

    if (_RESULTS_DIR / "manipulation_analysis" / "manipulation_results.json").exists():
        reports.append({
            "id": "manipulation_analysis",
            "name": "Manipulation-Wise Breakdown Report",
            "experiment": "resnet50_regularized",
            "status": "Ready for review",
            "has_results": True,
        })

    return reports


def export_experiment_report(experiment_id: str) -> Optional[str]:
    """Generate a markdown report for an experiment from its metrics."""
    detail = load_experiment_detail(experiment_id)
    if detail is None:
        return None

    name = _MODEL_DISPLAY_NAMES.get(experiment_id, experiment_id)
    tm = detail.get("detailed_metrics", detail.get("test_metrics", {}))
    cm = detail.get("confusion_matrix", {})

    lines = [
        f"# {name} — Evaluation Report",
        "",
        f"**Experiment ID:** `{experiment_id}`",
        f"**Architecture:** {detail.get('architecture', '—')}",
        f"**Pre-trained:** {'Yes' if detail.get('pretrained') else 'No'}",
        "",
        "## Test Metrics",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    for key, val in tm.items():
        if isinstance(val, float):
            lines.append(f"| {key} | {val:.4f} |")
        else:
            lines.append(f"| {key} | {val} |")

    if cm:
        lines.extend([
            "",
            "## Confusion Matrix",
            "",
            "|  | Predicted REAL | Predicted FAKE |",
            "| --- | --- | --- |",
            f"| **Actual REAL** | TN = {cm.get('tn', '—')} | FP = {cm.get('fp', '—')} |",
            f"| **Actual FAKE** | FN = {cm.get('fn', '—')} | TP = {cm.get('tp', '—')} |",
        ])

    lines.extend(["", f"---", "", f"*Exported from DeepShield-X*"])
    return "\n".join(lines)


def export_analysis_report(report_id: str) -> Optional[str]:
    """Render the stored robustness/manipulation JSON as a real markdown export."""
    if report_id == "robustness_analysis":
        rows = load_robustness_results()
        title = "Robustness Stress-Test Report"
        columns = [("Perturbation", "perturbation_name"), ("Samples", "evaluated_samples"), ("Recall", "detection_rate_recall"), ("Retention %", "performance_retention_pct")]
    elif report_id == "manipulation_analysis":
        rows = load_manipulation_results()
        title = "Manipulation-Wise Breakdown Report"
        columns = [("Manipulation", "manipulation"), ("Samples", "sample_count"), ("Accuracy", "accuracy"), ("F1", "f1")]
    else:
        return None
    if not rows:
        return None
    lines = [f"# {title}", "", "| " + " | ".join(column[0] for column in columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        values = []
        for _, key in columns:
            value = row.get(key, "—")
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", "---", "", "*Generated from DeepShield-X evaluation artifacts.*"])
    return "\n".join(lines)
