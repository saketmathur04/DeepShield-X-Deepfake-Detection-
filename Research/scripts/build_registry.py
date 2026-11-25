import json
from pathlib import Path
import math

def build_registry(root_dir: Path):
    results_dir = root_dir / "experiments" / "results"
    registry_json_path = root_dir / "experiments" / "experiment_registry.json"
    registry_md_path = root_dir / "experiments" / "experiment_registry.md"

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
    
    registry = []
    
    for model_id in expected_models:
        model_dir = results_dir / model_id
        
        metrics_file = model_dir / "evaluation" / "test" / "metrics.json"
        fallback_metrics = model_dir / f"{model_id}_metrics.json"
        
        metrics_data = None
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                metrics_data = json.load(f)
        elif fallback_metrics.exists():
            with open(fallback_metrics, "r") as f:
                metrics_data = json.load(f)
        else:
            # Glob fallback: find any *_metrics.json in the model directory
            candidates = list(model_dir.glob("*_metrics.json"))
            if candidates:
                with open(candidates[0], "r") as f:
                    metrics_data = json.load(f)
                
        if metrics_data is None:
            continue
            
        actual_metrics = metrics_data.get("metrics", metrics_data)
        
        # Determine specific details based on model string
        is_pretrained = model_id not in ["custom_cnn", "class_balanced_custom_cnn"]
        is_balanced = "balanced" in model_id
        
        notes = "Baseline model."
        if model_id == "class_balanced_custom_cnn":
            notes = "Controlled experiment: tests effect of class weighting."
        elif model_id == "resnet50_regularized":
            notes = "Controlled experiment: applies strong regularization to combat overfitting observed in ResNet50 baseline."
            
        entry = {
            "experiment_id": model_id,
            "architecture": model_id.split("_regularized")[0].replace("class_balanced_", ""),
            "pretrained": is_pretrained,
            "pos_weight": is_balanced,
            "test_metrics": {
                "accuracy": actual_metrics.get("accuracy", 0),
                "precision": actual_metrics.get("precision", 0),
                "recall": actual_metrics.get("recall", 0),
                "f1": actual_metrics.get("f1", 0),
                "roc_auc": actual_metrics.get("roc_auc", 0),
                "pr_auc": actual_metrics.get("pr_auc", 0),
                "specificity": actual_metrics.get("specificity", 0),
                "balanced_accuracy": actual_metrics.get("balanced_accuracy", 0),
                "mcc": actual_metrics.get("mcc", 0)
            },
            "notes": notes
        }
        registry.append(entry)
        
    with open(registry_json_path, "w") as f:
        json.dump(registry, f, indent=4)
        
    # Generate MD
    md_content = ["# DeepShield-X Experiment Registry\n"]
    md_content.append("| Experiment ID | Architecture | Pretrained | Balanced | F1 | ROC-AUC | PR-AUC | Specificity | Notes |")
    md_content.append("|---|---|---|---|---|---|---|---|---|")
    for r in registry:
        m = r["test_metrics"]
        md_content.append(
            f"| {r['experiment_id']} | {r['architecture']} | {r['pretrained']} | {r['pos_weight']} | "
            f"{m['f1']:.4f} | {m['roc_auc']:.4f} | {m['pr_auc']:.4f} | {m['specificity']:.4f} | {r['notes']} |"
        )
        
    with open(registry_md_path, "w") as f:
        f.write("\n".join(md_content))
        
    print(f"Registry built: {len(registry)} experiments recorded.")

if __name__ == "__main__":
    build_registry(Path(__file__).resolve().parent.parent)
