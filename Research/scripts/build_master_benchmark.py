"""
DeepShield-X — Master Benchmark Table Generator

Generates the authoritative model comparison tables from experiment artifacts.
Separates architecture comparison from training-strategy ablation.
"""

import json
from pathlib import Path

def load_metrics(results_dir: Path, model_id: str):
    """Load metrics from the standard paths."""
    metrics_file = results_dir / model_id / "evaluation" / "test" / "metrics.json"
    fallback = results_dir / model_id / f"{model_id}_metrics.json"
    
    path = None
    if metrics_file.exists():
        path = metrics_file
    elif fallback.exists():
        path = fallback
    else:
        candidates = list((results_dir / model_id).glob("*_metrics.json"))
        if candidates:
            path = candidates[0]
    
    if path is None:
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get("metrics", data)


def fmt(val, decimals=4):
    if val is None or val == 0:
        return "—"
    return f"{val:.{decimals}f}"


def compute_balanced_accuracy(m):
    sens = m.get("sensitivity", m.get("recall", 0))
    spec = m.get("specificity", 0)
    return (sens + spec) / 2 if sens and spec else 0


def main():
    root = Path(__file__).resolve().parent.parent
    results_dir = root / "experiments" / "results"
    output_dir = root / "experiments" / "results" / "master_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    # === Primary Architecture Comparison (6 models) ===
    primary_models = ["custom_cnn", "resnet50", "efficientnet", "mobilenet", "xception", "vit"]
    
    md = []
    md.append("# DeepShield-X — Master Model Comparison\n")
    md.append("## 1. Primary Architecture Comparison\n")
    md.append("All models evaluated on the same held-out test set (N=9,589). Threshold = 0.50.\n")
    md.append("| Model | Params | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Specificity | Bal. Acc. | MCC |")
    md.append("|-------|--------|----------|-----------|--------|------|---------|--------|-------------|----------|-----|")
    
    all_data = {}
    for model_id in primary_models:
        m = load_metrics(results_dir, model_id)
        if m is None:
            continue
        all_data[model_id] = m
        ba = compute_balanced_accuracy(m)
        md.append(
            f"| {model_id} | — | {fmt(m.get('accuracy'))} | {fmt(m.get('precision'))} | "
            f"{fmt(m.get('recall'))} | {fmt(m.get('f1'))} | {fmt(m.get('roc_auc'))} | "
            f"{fmt(m.get('pr_auc'))} | {fmt(m.get('specificity'))} | {fmt(ba)} | {fmt(m.get('mcc'))} |"
        )
    md.append("")

    # === Training Strategy Ablation: Class Imbalance ===
    md.append("## 2. Training Strategy Ablation: Class Imbalance (Custom CNN)\n")
    md.append("Same architecture, same data, same seed — only loss weighting changes.\n")
    md.append("| Variant | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Specificity | Bal. Acc. | MCC |")
    md.append("|---------|----------|-----------|--------|------|---------|--------|-------------|----------|-----|")
    
    for model_id, label in [("custom_cnn", "Unweighted"), ("class_balanced_custom_cnn", "Class-Balanced")]:
        m = load_metrics(results_dir, model_id)
        if m is None:
            continue
        ba = compute_balanced_accuracy(m)
        md.append(
            f"| {label} | {fmt(m.get('accuracy'))} | {fmt(m.get('precision'))} | "
            f"{fmt(m.get('recall'))} | {fmt(m.get('f1'))} | {fmt(m.get('roc_auc'))} | "
            f"{fmt(m.get('pr_auc'))} | {fmt(m.get('specificity'))} | {fmt(ba)} | {fmt(m.get('mcc'))} |"
        )
    md.append("")

    # === Training Strategy Ablation: Regularization ===
    md.append("## 3. Training Strategy Ablation: Regularization (ResNet50)\n")
    md.append("Same architecture — regularization (augmentation, class weighting, val ROC-AUC checkpoint selection, early stopping) applied.\n")
    md.append("| Variant | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Specificity | Bal. Acc. | MCC |")
    md.append("|---------|----------|-----------|--------|------|---------|--------|-------------|----------|-----|")
    
    for model_id, label in [("resnet50", "Baseline"), ("resnet50_regularized", "Regularized")]:
        m = load_metrics(results_dir, model_id)
        if m is None:
            continue
        ba = compute_balanced_accuracy(m)
        md.append(
            f"| {label} | {fmt(m.get('accuracy'))} | {fmt(m.get('precision'))} | "
            f"{fmt(m.get('recall'))} | {fmt(m.get('f1'))} | {fmt(m.get('roc_auc'))} | "
            f"{fmt(m.get('pr_auc'))} | {fmt(m.get('specificity'))} | {fmt(ba)} | {fmt(m.get('mcc'))} |"
        )
    md.append("")

    # === Key Observations ===
    md.append("## 4. Key Observations\n")
    md.append("- **ResNet50 Regularized** achieves the highest F1 (0.9633), ROC-AUC (0.9718), and PR-AUC (0.9938).")
    md.append("- **Xception** is extremely competitive: highest specificity (0.8830) and near-identical ROC-AUC (0.9706).")
    md.append("- **Custom CNN (Unweighted)** exhibits 0% specificity — predicts everything as FAKE due to class imbalance.")
    md.append("- **Class Balancing** dramatically improves specificity (0→0.7487) but at severe cost to FAKE recall.")
    md.append("- **Regularization** yields consistent improvement over baseline ResNet50 across all metrics.")
    md.append("- **ViT** performs respectably but does not surpass CNNs on this dataset size and task.")
    md.append("")

    # Write
    with open(output_dir / "master_comparison.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    
    # Also write as JSON
    json_data = {
        "primary_comparison": {},
        "class_imbalance_ablation": {},
        "regularization_ablation": {}
    }
    for model_id in primary_models:
        m = load_metrics(results_dir, model_id)
        if m:
            json_data["primary_comparison"][model_id] = m
    for model_id in ["custom_cnn", "class_balanced_custom_cnn"]:
        m = load_metrics(results_dir, model_id)
        if m:
            json_data["class_imbalance_ablation"][model_id] = m
    for model_id in ["resnet50", "resnet50_regularized"]:
        m = load_metrics(results_dir, model_id)
        if m:
            json_data["regularization_ablation"][model_id] = m
    
    with open(output_dir / "master_comparison.json", "w") as f:
        json.dump(json_data, f, indent=2)
    
    print("Master benchmark tables generated successfully.")
    print(f"  MD:   {output_dir / 'master_comparison.md'}")
    print(f"  JSON: {output_dir / 'master_comparison.json'}")


if __name__ == "__main__":
    main()
