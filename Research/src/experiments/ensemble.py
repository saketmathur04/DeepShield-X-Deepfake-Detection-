"""
DeepShield-X — Ensemble Experiment (Experiment 8)

Soft-voting ensemble of multiple models.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


def ensemble_predict(
    models: List[nn.Module],
    dataloader: DataLoader,
    device: torch.device,
    method: str = "soft_voting",
) -> Dict[str, np.ndarray]:
    """
    Ensemble prediction using soft voting (probability averaging).

    Args:
        models: List of trained models.
        dataloader: Test data loader.
        device: Compute device.
        method: Ensemble method ("soft_voting").

    Returns:
        Dict with y_true, y_pred, y_prob arrays.
    """
    all_probs = []
    all_labels = []

    for model in models:
        model.to(device)
        model.eval()

    # Collect predictions from each model
    model_probs = [[] for _ in models]

    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device)

            for i, model in enumerate(models):
                outputs = model(images).squeeze(-1)
                probs = torch.sigmoid(outputs).cpu().numpy()
                model_probs[i].extend(probs)

            all_labels.extend(labels.numpy())

    # Stack and average
    model_probs = [np.array(p) for p in model_probs]
    y_true = np.array(all_labels)

    if method == "soft_voting":
        y_prob = np.mean(model_probs, axis=0)
    else:
        y_prob = np.mean(model_probs, axis=0)

    y_pred = (y_prob >= 0.5).astype(int)

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "individual_probs": model_probs,
    }


def evaluate_ensemble(
    models: List[nn.Module],
    model_names: List[str],
    dataloader: DataLoader,
    device: torch.device,
    output_dir: Optional[str] = None,
) -> Dict:
    """Run ensemble evaluation and compare with individual models."""
    import json
    from pathlib import Path

    results = {}

    # Individual model evaluations
    for model, name in zip(models, model_names):
        preds = ensemble_predict([model], dataloader, device)
        metrics = compute_all_metrics(preds["y_true"], preds["y_pred"], preds["y_prob"])
        results[name] = metrics

    # Ensemble evaluation
    ensemble_preds = ensemble_predict(models, dataloader, device)
    ensemble_metrics = compute_all_metrics(
        ensemble_preds["y_true"], ensemble_preds["y_pred"], ensemble_preds["y_prob"]
    )
    results["ensemble"] = ensemble_metrics

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / "ensemble_results.json", "w") as f:
            json.dump(results, f, indent=2)

    return results
