"""
DeepShield-X — Robustness Evaluation

Test model robustness against JPEG compression, Gaussian blur, and noise.
Applies degradations dynamically — never modifies the original test data.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.data.preprocessing import (
    apply_gaussian_blur,
    apply_gaussian_noise,
    apply_jpeg_compression,
)
from src.evaluation.evaluator import Evaluator
from src.evaluation.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


class DegradedDataset(Dataset):
    """
    Wraps an existing dataset and applies a degradation function.
    Original data is never modified — degradation is applied on-the-fly.
    """

    def __init__(self, base_dataset: Dataset, degradation_fn, degradation_name: str = ""):
        self.base_dataset = base_dataset
        self.degradation_fn = degradation_fn
        self.degradation_name = degradation_name

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label, metadata = self.base_dataset[idx]

        # Convert tensor to numpy, apply degradation, convert back
        if isinstance(image, torch.Tensor):
            img_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        else:
            img_np = image

        degraded = self.degradation_fn(img_np)

        if isinstance(image, torch.Tensor):
            degraded = torch.from_numpy(degraded).permute(2, 0, 1).float() / 255.0

        metadata["degradation"] = self.degradation_name
        return degraded, label, metadata


def evaluate_jpeg_robustness(
    model: nn.Module,
    test_dataset: Dataset,
    device: torch.device,
    qualities: List[int] = None,
    batch_size: int = 32,
    output_dir: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model robustness to JPEG compression.

    Args:
        model: Trained model.
        test_dataset: Original test dataset.
        device: Compute device.
        qualities: JPEG quality levels to test.
        batch_size: Batch size.
        output_dir: Directory to save results.

    Returns:
        Dict mapping quality level → metrics.
    """
    if qualities is None:
        qualities = [100, 80, 60, 40, 20]

    results = {}
    evaluator = Evaluator(model, device)

    for quality in qualities:
        logger.info(f"Evaluating JPEG quality={quality}")

        degraded_ds = DegradedDataset(
            test_dataset,
            lambda img, q=quality: apply_jpeg_compression(img, q),
            degradation_name=f"jpeg_q{quality}",
        )

        from src.data.dataset import create_dataloader
        loader = create_dataloader(degraded_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        eval_result = evaluator.evaluate(loader, model_name=f"jpeg_q{quality}")
        results[f"q{quality}"] = eval_result["metrics"]

    if output_dir:
        import json
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / "jpeg_robustness.json", "w") as f:
            json.dump(results, f, indent=2)

    return results


def evaluate_blur_robustness(
    model: nn.Module,
    test_dataset: Dataset,
    device: torch.device,
    severities: List[int] = None,
    batch_size: int = 32,
    output_dir: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Evaluate model robustness to Gaussian blur."""
    if severities is None:
        severities = [1, 3, 5, 7, 9]

    results = {}
    evaluator = Evaluator(model, device)

    for severity in severities:
        logger.info(f"Evaluating blur severity={severity}")

        degraded_ds = DegradedDataset(
            test_dataset,
            lambda img, s=severity: apply_gaussian_blur(img, s),
            degradation_name=f"blur_s{severity}",
        )

        from src.data.dataset import create_dataloader
        loader = create_dataloader(degraded_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        eval_result = evaluator.evaluate(loader, model_name=f"blur_s{severity}")
        results[f"s{severity}"] = eval_result["metrics"]

    if output_dir:
        import json
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / "blur_robustness.json", "w") as f:
            json.dump(results, f, indent=2)

    return results


def evaluate_noise_robustness(
    model: nn.Module,
    test_dataset: Dataset,
    device: torch.device,
    severities: List[float] = None,
    batch_size: int = 32,
    output_dir: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Evaluate model robustness to Gaussian noise."""
    if severities is None:
        severities = [0.01, 0.03, 0.05, 0.1, 0.15]

    results = {}
    evaluator = Evaluator(model, device)

    for sigma in severities:
        logger.info(f"Evaluating noise sigma={sigma}")

        degraded_ds = DegradedDataset(
            test_dataset,
            lambda img, s=sigma: apply_gaussian_noise(img, s),
            degradation_name=f"noise_s{sigma}",
        )

        from src.data.dataset import create_dataloader
        loader = create_dataloader(degraded_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        eval_result = evaluator.evaluate(loader, model_name=f"noise_s{sigma}")
        results[f"s{sigma}"] = eval_result["metrics"]

    if output_dir:
        import json
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / "noise_robustness.json", "w") as f:
            json.dump(results, f, indent=2)

    return results
