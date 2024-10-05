from fastapi import APIRouter, HTTPException
from backend.services.data_loader import (
    load_robustness_results,
    load_manipulation_results,
    load_error_analysis,
    load_experiment_registry,
    load_master_comparison,
    load_bootstrap_results,
    load_system_info,
    load_research_plots,
)

router = APIRouter(prefix="/api/analysis", tags=["analysis"])

@router.get("/robustness")
def get_robustness():
    """Get robustness evaluation results."""
    return load_robustness_results()

@router.get("/manipulation")
def get_manipulation():
    """Get manipulation-wise evaluation results."""
    return load_manipulation_results()

@router.get("/errors/{experiment_id}")
def get_errors(experiment_id: str):
    """Get error analysis summary for a specific experiment."""
    errors = load_error_analysis(experiment_id)
    if not errors:
        raise HTTPException(status_code=404, detail="Error analysis not found")
    return errors

@router.get("/comparison")
def get_comparison():
    """Aggregate results for side-by-side comparison."""
    registry = load_experiment_registry()
    
    comparison = []
    for exp in registry:
        eid = exp["experiment_id"]
        tm = exp.get("test_metrics", {})
        
        comparison.append({
            "id": eid,
            "name": eid.replace("_", " ").title().replace("Cnn", "CNN"),
            "architecture": exp.get("architecture", eid),
            "pretrained": exp.get("pretrained", False),
            "metrics": {
                "accuracy": tm.get("accuracy"),
                "f1": tm.get("f1"),
                "rocAuc": tm.get("roc_auc"),
                "prAuc": tm.get("pr_auc"),
                "specificity": tm.get("specificity"),
                "balancedAccuracy": tm.get("balanced_accuracy"),
                "mcc": tm.get("mcc")
            }
        })
    return comparison

@router.get("/master-benchmark")
def get_master_benchmark():
    """Get master benchmark comparing all models on frozen held-out test partition."""
    return load_master_comparison()

@router.get("/statistical")
def get_statistical():
    """Get bootstrap statistical confidence analysis (1,000 iterations, 95% CI)."""
    return load_bootstrap_results()

@router.get("/system-info")
def get_system_telemetry():
    """Get live runtime system status, hardware devices, and model checkpoint counts."""
    return load_system_info()

@router.get("/research-plots")
def get_research_plots():
    """Get all scientific visualization figures generated during research."""
    return load_research_plots()
