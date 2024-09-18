"""
DeepShield-X — Evaluator

Complete evaluation pipeline: run inference, compute metrics, generate plots.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
try:
    from torch.amp import autocast as _autocast
    def get_autocast(device_type="cuda"):
        return _autocast(device_type)
except (ImportError, AttributeError):
    from torch.cuda.amp import autocast as _autocast
    def get_autocast(device_type="cuda"):
        return _autocast()
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.metrics import (
    compute_all_metrics,
    compute_pr_curve,
    compute_roc_curve,
    format_metrics_table,
)

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Model evaluation framework.

    Runs inference on a test set, computes all metrics, and saves results.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        threshold: float = 0.5,
        use_amp: bool = True,
    ):
        self.model = model.to(device)
        self.device = device
        self.threshold = threshold
        self.use_amp = use_amp and device.type == "cuda"

    @torch.no_grad()
    def predict(self, dataloader: DataLoader) -> Dict[str, np.ndarray]:
        """
        Run inference on a dataloader.

        Returns:
            Dict with y_true, y_pred, y_prob arrays and metadata.
        """
        self.model.eval()
        all_labels = []
        all_probs = []
        all_metadata = []

        pbar = tqdm(dataloader, desc="Evaluating", leave=False)
        for images, labels, metadata in pbar:
            images = images.to(self.device, non_blocking=True)

            if self.use_amp:
                with get_autocast(self.device.type):
                    outputs = self.model(images).squeeze(-1)
            else:
                outputs = self.model(images).squeeze(-1)

            probs = torch.sigmoid(outputs).cpu().numpy()
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)
            all_metadata.extend(metadata)

        y_true = np.array(all_labels)
        y_prob = np.array(all_probs)
        y_pred = (y_prob >= self.threshold).astype(int)

        return {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "metadata": all_metadata,
        }

    def evaluate(
        self,
        dataloader: DataLoader,
        model_name: str = "model",
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Full evaluation: inference → metrics → save results.

        Args:
            dataloader: Test data loader.
            model_name: Model name for result labeling.
            output_dir: Directory to save results.

        Returns:
            Dict with metrics, predictions, and timing.
        """
        start_time = time.time()

        # Run inference
        predictions = self.predict(dataloader)
        inference_time = time.time() - start_time

        # Compute metrics
        metrics = compute_all_metrics(
            y_true=predictions["y_true"],
            y_pred=predictions["y_pred"],
            y_prob=predictions["y_prob"],
            threshold=self.threshold,
        )

        # Compute curves
        fpr, tpr, roc_thresholds = compute_roc_curve(
            predictions["y_true"], predictions["y_prob"]
        )
        precision_curve, recall_curve, pr_thresholds = compute_pr_curve(
            predictions["y_true"], predictions["y_prob"]
        )

        # Timing info
        num_samples = len(predictions["y_true"])
        avg_inference_ms = (inference_time / num_samples * 1000) if num_samples > 0 else 0

        result = {
            "model_name": model_name,
            "metrics": metrics,
            "predictions": predictions,
            "curves": {
                "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
                "pr": {"precision": precision_curve.tolist(), "recall": recall_curve.tolist()},
            },
            "timing": {
                "total_seconds": round(inference_time, 3),
                "avg_per_sample_ms": round(avg_inference_ms, 3),
                "num_samples": num_samples,
            },
        }

        # Log results
        logger.info(f"\n{format_metrics_table(metrics, model_name)}")
        logger.info(
            f"Inference: {num_samples} samples in {inference_time:.2f}s "
            f"({avg_inference_ms:.2f} ms/sample)"
        )

        # Save results
        if output_dir:
            self._save_results(result, output_dir, model_name)

        return result

    def _save_results(self, result: Dict, output_dir: str, model_name: str) -> None:
        """Save evaluation results to disk."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Save metrics JSON
        metrics_file = out_path / f"{model_name}_metrics.json"
        save_data = {
            "model_name": result["model_name"],
            "metrics": result["metrics"],
            "timing": result["timing"],
        }
        with open(metrics_file, "w") as f:
            json.dump(save_data, f, indent=2)

        # Save metrics markdown
        md_file = out_path / f"{model_name}_metrics.md"
        md_content = format_metrics_table(result["metrics"], model_name)
        with open(md_file, "w") as f:
            f.write(md_content)

        logger.info(f"Results saved to: {out_path}")
