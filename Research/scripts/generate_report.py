"""
DeepShield-X — Report Generator

Aggregates experiment results into a markdown report.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logging import setup_logging

def discover_experiments(results_dir: Path):
    """Discover experiments and build metadata records. Expects the standard 8 models."""
    expected_models = [
        "custom_cnn",
        "class_balanced_custom_cnn",
        "resnet50",
        "resnet50_regularized",
        "efficientnet",
        "mobilenet",
        "xception",
        "vit"
    ]
    
    records = []
    for model_id in expected_models:
        metrics_file = results_dir / model_id / "evaluation" / "test" / "metrics.json"
        
        # some baseline models might have slightly different paths according to my audit (e.g. custom_cnn_metrics.json directly in folder)
        fallback_metrics = results_dir / model_id / f"{model_id}_metrics.json"
        
        available = False
        metrics_data = {}
        if metrics_file.exists():
            available = True
            with open(metrics_file, "r") as f:
                metrics_data = json.load(f)
        elif fallback_metrics.exists():
            available = True
            with open(fallback_metrics, "r") as f:
                metrics_data = json.load(f)
        
        records.append({
            "id": model_id,
            "available": available,
            "metrics": metrics_data.get("metrics", metrics_data)  # Some JSONs have a "metrics" wrapper, some don't
        })
        
    return records


def build_report(results_dir: Path):
    """Builds the comprehensive markdown report string."""
    report = []
    report.append("# DeepShield-X — Experiment Report")
    report.append(f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

    records = discover_experiments(results_dir)

    report.append("## 6. Primary Architecture Comparison\n")
    report.append("| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |")
    report.append("|-------|----------|-----------|--------|-----|---------|")
    
    for record in records:
        if record["available"]:
            metrics = record["metrics"]
            name = record["id"]
            report.append(
                f"| {name} | {metrics.get('accuracy', 0):.4f} | "
                f"{metrics.get('precision', 0):.4f} | {metrics.get('recall', 0):.4f} | "
                f"{metrics.get('f1', 0):.4f} | {metrics.get('roc_auc', 0):.4f} |"
            )
    report.append("")

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="DeepShield-X Report Generator")
    parser.add_argument("--results-dir", type=str, default="./experiments/results")
    parser.add_argument("--output", type=str, default="./experiments/results/report.md")
    args = parser.parse_args()
    
    logger = setup_logging("deepshield")
    results_dir = Path(args.results_dir)
    
    report_content = build_report(results_dir)
    
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.write(report_content)

    logger.info(f"Report generated: {output}")


if __name__ == "__main__":
    main()
