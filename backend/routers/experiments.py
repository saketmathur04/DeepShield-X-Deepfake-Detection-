from fastapi import APIRouter, HTTPException
from backend.services.data_loader import load_experiment_registry, load_experiment_detail

router = APIRouter(prefix="/api/experiments", tags=["experiments"])

@router.get("")
def list_experiments():
    """List all experiments from the registry."""
    # We want to format this closely to what the frontend expects
    registry = load_experiment_registry()
    
    experiments = []
    for exp in registry:
        eid = exp["experiment_id"]
        
        # Determine status
        status = "Not Run"
        if exp.get("test_metrics", {}).get("accuracy", 0) > 0:
            status = "Completed"
            
        experiments.append({
            "id": eid,
            "name": eid.replace("_", " ").title().replace("Cnn", "CNN"),
            "model": exp.get("architecture", eid),
            "dataset": "FaceForensics++ C23",
            "status": status,
            "metrics": {
                "accuracy": exp.get("test_metrics", {}).get("accuracy"),
                "f1": exp.get("test_metrics", {}).get("f1"),
                "rocAuc": exp.get("test_metrics", {}).get("roc_auc"),
            }
        })
    return experiments

@router.get("/{experiment_id}")
def get_experiment(experiment_id: str):
    """Get detailed results for an experiment."""
    detail = load_experiment_detail(experiment_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
        
    metrics = detail.get("detailed_metrics", detail.get("test_metrics", {}))
    
    # Map metrics to camelCase for the frontend
    mapped_metrics = {
        "accuracy": metrics.get("accuracy"),
        "balancedAccuracy": metrics.get("balanced_accuracy"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "sensitivity": metrics.get("sensitivity", metrics.get("recall")),
        "specificity": metrics.get("specificity"),
        "mcc": metrics.get("mcc"),
        "rocAuc": metrics.get("roc_auc"),
        "prAuc": metrics.get("pr_auc"),
    }
    
    return {
        "id": detail["experiment_id"],
        "name": detail["experiment_id"].replace("_", " ").title().replace("Cnn", "CNN"),
        "model": detail.get("architecture", detail["experiment_id"]),
        "dataset": "FaceForensics++ C23",
        "status": "Completed" if metrics.get("accuracy", 0) > 0 else "Not Run",
        "metrics": mapped_metrics,
        "config": detail.get("config", {}),
        "confusion": detail.get("confusion_matrix"),
    }
