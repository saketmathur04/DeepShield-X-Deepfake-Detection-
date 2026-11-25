#!/usr/bin/env python3
"""
DeepShield-X — Phase 4/5/6 EC2 Runner
=====================================
Run this script ON THE EC2 INSTANCE where the dataset and GPU are available.

This script executes:
  Phase 4: Robustness Study (JPEG, blur, noise, resolution)
  Phase 5: Grad-CAM Explainability
  Phase 6: Efficiency Analysis (param counts, inference times)

Usage:
  cd ~/DeepShield-X
  source /opt/pytorch/bin/activate
  python scripts/run_phase456.py
"""

import json
import sys
import time
import csv
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.device import get_device
from src.utils.logging import setup_logging
from src.models.factory import create_model, get_model_info, MODEL_REGISTRY
from src.data.dataset import DeepFakeDataset, create_dataloader
from src.data.preprocessing import get_val_transforms
from src.evaluation.evaluator import Evaluator
from src.evaluation.robustness import (
    evaluate_jpeg_robustness,
    evaluate_blur_robustness,
    evaluate_noise_robustness,
)

logger = setup_logging("deepshield", "INFO")

# ============================================================
# CONFIG
# ============================================================
SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 32
STRONGEST_MODEL = "resnet50"
STRONGEST_CHECKPOINT = "experiments/checkpoints/resnet50_regularized/resnet50_best.pth"
STRONGEST_EXPERIMENT_ID = "resnet50_regularized"

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "experiments" / "results"
ROBUSTNESS_DIR = RESULTS_DIR / "robustness"
EXPLAINABILITY_DIR = RESULTS_DIR / "explainability"
EFFICIENCY_DIR = RESULTS_DIR / "efficiency"


def load_test_dataset():
    """Load the test dataset from splits."""
    cfg = load_config("configs/default.yaml")
    splits_dir = ROOT / "data" / "splits"
    test_csv = splits_dir / "test.csv"
    root_dir = str(ROOT / "data" / "faces")
    val_transform = get_val_transforms(IMAGE_SIZE)
    test_dataset = DeepFakeDataset(str(test_csv), root_dir, val_transform, IMAGE_SIZE)
    logger.info(f"Test dataset loaded: {len(test_dataset)} samples")
    return test_dataset


def load_strongest_model(device):
    """Load the strongest model (ResNet50 Regularized)."""
    from src.utils.checkpoint import load_checkpoint
    model = create_model(STRONGEST_MODEL, pretrained=False)
    ckpt_path = ROOT / STRONGEST_CHECKPOINT
    load_checkpoint(str(ckpt_path), model, device=device)
    model.to(device)
    model.eval()
    logger.info(f"Loaded strongest model from {ckpt_path}")
    return model


# ============================================================
# PHASE 4: ROBUSTNESS
# ============================================================
def run_robustness(model, test_dataset, device):
    """Run complete robustness evaluation."""
    logger.info("=" * 60)
    logger.info("PHASE 4: ROBUSTNESS STUDY")
    logger.info("=" * 60)

    output_dir = str(ROBUSTNESS_DIR)

    # 4a. JPEG Compression
    logger.info("--- JPEG Compression Robustness ---")
    jpeg_results = evaluate_jpeg_robustness(
        model, test_dataset, device,
        qualities=[95, 75, 50, 25],
        batch_size=BATCH_SIZE,
        output_dir=output_dir,
    )

    # 4b. Gaussian Blur
    logger.info("--- Gaussian Blur Robustness ---")
    blur_results = evaluate_blur_robustness(
        model, test_dataset, device,
        severities=[1, 3, 5, 7],
        batch_size=BATCH_SIZE,
        output_dir=output_dir,
    )

    # 4c. Gaussian Noise
    logger.info("--- Gaussian Noise Robustness ---")
    noise_results = evaluate_noise_robustness(
        model, test_dataset, device,
        severities=[0.01, 0.05, 0.10, 0.20],
        batch_size=BATCH_SIZE,
        output_dir=output_dir,
    )

    # 4d. Combined CSV
    logger.info("--- Generating robustness_results.csv ---")
    rows = []
    
    # Get clean baseline
    evaluator = Evaluator(model, device)
    test_loader = create_dataloader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    clean_result = evaluator.evaluate(test_loader, model_name="clean_baseline")
    clean_m = clean_result["metrics"]
    rows.append({
        "perturbation": "clean", "severity": "none",
        "accuracy": clean_m["accuracy"], "f1": clean_m["f1"],
        "roc_auc": clean_m["roc_auc"], "pr_auc": clean_m["pr_auc"],
        "recall": clean_m["recall"], "specificity": clean_m["specificity"],
    })

    for key, m in jpeg_results.items():
        rows.append({
            "perturbation": "jpeg_compression", "severity": key,
            "accuracy": m["accuracy"], "f1": m["f1"],
            "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"],
            "recall": m["recall"], "specificity": m["specificity"],
        })
    for key, m in blur_results.items():
        rows.append({
            "perturbation": "gaussian_blur", "severity": key,
            "accuracy": m["accuracy"], "f1": m["f1"],
            "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"],
            "recall": m["recall"], "specificity": m["specificity"],
        })
    for key, m in noise_results.items():
        rows.append({
            "perturbation": "gaussian_noise", "severity": key,
            "accuracy": m["accuracy"], "f1": m["f1"],
            "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"],
            "recall": m["recall"], "specificity": m["specificity"],
        })

    csv_path = ROBUSTNESS_DIR / "robustness_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Robustness CSV saved: {csv_path}")
    logger.info("PHASE 4 COMPLETE")
    return rows


# ============================================================
# PHASE 5: EXPLAINABILITY
# ============================================================
def run_explainability(model, test_dataset, device):
    """Run Grad-CAM explainability on selected samples."""
    logger.info("=" * 60)
    logger.info("PHASE 5: EXPLAINABILITY (Grad-CAM)")
    logger.info("=" * 60)

    try:
        from src.explainability.gradcam import GradCAMExplainer, GRADCAM_AVAILABLE
        if not GRADCAM_AVAILABLE:
            logger.warning("pytorch-grad-cam not installed. Install: pip install pytorch-grad-cam")
            return
    except ImportError:
        logger.warning("Grad-CAM module not available.")
        return

    explainer = GradCAMExplainer(model, device=device)
    output_dir = EXPLAINABILITY_DIR / STRONGEST_EXPERIMENT_ID
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run predictions to find interesting samples
    evaluator = Evaluator(model, device)
    test_loader = create_dataloader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    result = evaluator.evaluate(test_loader, model_name=STRONGEST_EXPERIMENT_ID)
    
    predictions = result["predictions"]
    y_true = predictions["y_true"]
    y_pred = predictions["y_pred"]
    y_prob = predictions["y_prob"]

    # Categorize samples
    correct_real = np.where((y_true == 0) & (y_pred == 0))[0]
    correct_fake = np.where((y_true == 1) & (y_pred == 1))[0]
    false_positive = np.where((y_true == 0) & (y_pred == 1))[0]
    false_negative = np.where((y_true == 1) & (y_pred == 0))[0]

    # Pick high-confidence examples
    categories = {
        "correct_real": correct_real,
        "correct_fake": correct_fake,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }

    for category_name, indices in categories.items():
        if len(indices) == 0:
            logger.warning(f"No samples found for category: {category_name}")
            continue

        # Sort by confidence (probability distance from 0.5)
        confidences = np.abs(y_prob[indices] - 0.5)
        sorted_idx = indices[np.argsort(-confidences)]  # highest confidence first
        
        # Take up to 5 samples from each category
        n_samples = min(5, len(sorted_idx))
        for i in range(n_samples):
            idx = sorted_idx[i]
            image_tensor, label, metadata = test_dataset[idx]
            
            # Get original image for overlay
            original_image = image_tensor.permute(1, 2, 0).numpy()
            if original_image.max() > 1.0:
                original_image = original_image / 255.0

            explanation = explainer.explain(image_tensor, original_image)
            explainer.save_explanation(
                explanation,
                save_dir=str(output_dir / category_name),
                filename=f"sample_{i}",
                original_image=original_image,
            )

    logger.info(f"Explainability results saved: {output_dir}")
    logger.info("PHASE 5 COMPLETE")


# ============================================================
# PHASE 6: EFFICIENCY ANALYSIS
# ============================================================
def run_efficiency(device):
    """Compute parameter counts, model sizes, and inference times."""
    logger.info("=" * 60)
    logger.info("PHASE 6: EFFICIENCY ANALYSIS")
    logger.info("=" * 60)

    EFFICIENCY_DIR.mkdir(parents=True, exist_ok=True)
    
    model_names = ["custom_cnn", "resnet50", "efficientnet", "mobilenet", "xception", "vit"]
    results = {}

    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)

    for model_name in model_names:
        logger.info(f"Analyzing {model_name}...")
        model = create_model(model_name, pretrained=False)
        model.to(device)
        model.eval()
        
        info = get_model_info(model, model_name)

        # Inference timing (warmup + benchmark)
        with torch.no_grad():
            # Warmup
            for _ in range(10):
                _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()

            # Benchmark
            times = []
            for _ in range(100):
                start = time.perf_counter()
                _ = model(dummy_input)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)

        info["inference_ms"] = round(float(np.mean(times)), 3)
        info["inference_std_ms"] = round(float(np.std(times)), 3)

        # Checkpoint size
        ckpt_dir = ROOT / "experiments" / "checkpoints"
        ckpt_candidates = list(ckpt_dir.glob(f"**/{model_name}_best.pth"))
        if ckpt_candidates:
            info["checkpoint_size_mb"] = round(ckpt_candidates[0].stat().st_size / (1024*1024), 2)
        else:
            info["checkpoint_size_mb"] = None
            
        results[model_name] = info
        logger.info(f"  {model_name}: {info['total_params']:,} params, {info['inference_ms']:.1f}ms, {info['model_size_mb']}MB")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Save JSON
    with open(EFFICIENCY_DIR / "efficiency.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save CSV
    csv_path = EFFICIENCY_DIR / "model_efficiency.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "total_params", "trainable_params", "model_size_mb", 
                         "checkpoint_size_mb", "inference_ms", "inference_std_ms"])
        for name, info in results.items():
            writer.writerow([
                name, info["total_params"], info["trainable_params"],
                info["model_size_mb"], info.get("checkpoint_size_mb", ""),
                info["inference_ms"], info["inference_std_ms"]
            ])

    logger.info(f"Efficiency results saved: {EFFICIENCY_DIR}")
    logger.info("PHASE 6 COMPLETE")
    return results


# ============================================================
# MAIN
# ============================================================
def main():
    set_seed(SEED)
    device = get_device()
    logger.info(f"Device: {device}")

    # Load model and dataset
    model = load_strongest_model(device)
    test_dataset = load_test_dataset()

    # Phase 4: Robustness
    run_robustness(model, test_dataset, device)

    # Phase 5: Explainability
    run_explainability(model, test_dataset, device)

    # Free memory
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Phase 6: Efficiency
    run_efficiency(device)

    logger.info("=" * 60)
    logger.info("ALL PHASES (4, 5, 6) COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
