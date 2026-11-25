"""
DeepShield-X — Master Experiment Runner

Runs all experiments defined in configs/experiments.yaml.

Usage:
    python scripts/run_experiments.py --config configs/experiments.yaml
    python scripts/run_experiments.py --smoke-test
    python scripts/run_experiments.py --experiment model_comparison --smoke-test
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config, Config
from src.utils.seed import set_seed
from src.utils.device import get_device
from src.utils.logging import setup_logging
from src.models.factory import create_model, get_model_info, list_available_models
from src.data.dataset import SyntheticDataset, create_dataloader
from src.training.trainer import Trainer
from src.evaluation.evaluator import Evaluator
from src.evaluation.metrics import format_comparison_table
from src.utils.visualization import (
    plot_metric_comparison,
    plot_multi_metric_comparison,
    plot_robustness_curve,
    plot_efficiency_scatter,
)


def parse_args():
    parser = argparse.ArgumentParser(description="DeepShield-X Experiment Runner")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--experiments-config", type=str, default="configs/experiments.yaml")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Run a specific experiment only")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def run_model_comparison(cfg, exp_cfg, device, smoke_test=False, logger=None):
    """Experiment 1: Compare all model architectures."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 1: Model Comparison")
    logger.info("=" * 60)

    models = exp_cfg.get("models", list_available_models())
    output_dir = Path(exp_cfg.get("output_dir", "./experiments/results/model_comparison"))
    output_dir.mkdir(parents=True, exist_ok=True)

    image_size = cfg.get("dataset.image_size", 224)
    seed = cfg.get("project.seed", 42)
    all_results = {}

    for model_name in models:
        logger.info(f"\n--- Training: {model_name} ---")
        set_seed(seed)

        try:
            model = create_model(model_name, pretrained=not smoke_test if model_name != "custom_cnn" else False)
        except Exception as e:
            logger.warning(f"Skipping {model_name}: {e}")
            continue

        model_info = get_model_info(model, model_name)

        # Create datasets
        if smoke_test:
            train_ds = SyntheticDataset(50, image_size=64, seed=seed)
            val_ds = SyntheticDataset(20, image_size=64, seed=seed + 1)
            test_ds = SyntheticDataset(20, image_size=64, seed=seed + 2)
            batch_size = 8
        else:
            from src.data.dataset import DeepFakeDataset
            from src.data.preprocessing import get_train_transforms, get_val_transforms
            aug_dict = cfg.get("augmentation", {})
            if hasattr(aug_dict, 'to_dict'): aug_dict = aug_dict.to_dict()
            
            train_transform = get_train_transforms(image_size, aug_dict)
            val_transform = get_val_transforms(image_size)
            
            splits_dir = Path(cfg.get("dataset.root", "./data")) / "splits"
            root_dir = str(Path(cfg.get("dataset.root", "./data")))
            
            if not (splits_dir / "train.csv").exists():
                logger.error("Splits not found. Run prepare script first.")
                continue
                
            train_ds = DeepFakeDataset(str(splits_dir / "train.csv"), root_dir, train_transform, image_size)
            val_ds = DeepFakeDataset(str(splits_dir / "val.csv"), root_dir, val_transform, image_size)
            test_ds = DeepFakeDataset(str(splits_dir / "test.csv"), root_dir, val_transform, image_size)
            batch_size = cfg.get("training.batch_size", 32)

        train_loader = create_dataloader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0 if smoke_test else cfg.get("dataset.num_workers", 4))
        val_loader = create_dataloader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0 if smoke_test else cfg.get("dataset.num_workers", 4))
        test_loader = create_dataloader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0 if smoke_test else cfg.get("dataset.num_workers", 4))

        # Train
        train_cfg = cfg.to_dict() if hasattr(cfg, 'to_dict') else cfg
        trainer = Trainer(model, device, train_cfg)
        train_result = trainer.train(train_loader, val_loader, model_name, "model_comparison")

        # Evaluate
        evaluator = Evaluator(model, device)
        eval_result = evaluator.evaluate(test_loader, model_name, str(output_dir))

        all_results[model_name] = {
            **eval_result["metrics"],
            **model_info,
            "inference_ms": eval_result["timing"]["avg_per_sample_ms"],
        }

    # Save comparison
    if all_results:
        # Save JSON
        with open(output_dir / "comparison.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        # Save markdown table
        md_table = format_comparison_table(all_results)
        with open(output_dir / "comparison.md", "w") as f:
            f.write("# Model Comparison Results\n\n")
            f.write(md_table)

        # Generate plots
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        for metric in ["accuracy", "f1", "roc_auc", "precision", "recall"]:
            values = {m: r.get(metric, 0) for m, r in all_results.items()}
            plot_metric_comparison(
                values, metric,
                save_path=str(plots_dir / f"{metric}_comparison.png"),
            )

        plot_multi_metric_comparison(
            all_results,
            metrics=["accuracy", "precision", "recall", "f1"],
            title="Model Performance Comparison",
            save_path=str(plots_dir / "multi_metric_comparison.png"),
        )

        logger.info(f"Model comparison results saved to: {output_dir}")

    return all_results


def run_efficiency_experiment(cfg, exp_cfg, device, smoke_test=False, logger=None):
    """Experiment 7: Model efficiency analysis."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 7: Model Efficiency")
    logger.info("=" * 60)

    models = exp_cfg.get("models", list_available_models())
    output_dir = Path(exp_cfg.get("output_dir", "./experiments/results/efficiency"))
    output_dir.mkdir(parents=True, exist_ok=True)

    image_size = 64 if smoke_test else cfg.get("dataset.image_size", 224)
    results = {}

    for model_name in models:
        try:
            model = create_model(model_name, pretrained=False)
            model.to(device)
            model.eval()
        except Exception as e:
            logger.warning(f"Skipping {model_name}: {e}")
            continue

        info = get_model_info(model, model_name)

        # Measure inference time
        dummy = torch.randn(1, 3, image_size, image_size).to(device)
        num_warmup = exp_cfg.get("num_warmup", 5)
        num_iter = exp_cfg.get("num_iterations", 20) if smoke_test else exp_cfg.get("num_iterations", 100)

        with torch.no_grad():
            for _ in range(num_warmup):
                model(dummy)

            import time as time_module
            start = time_module.time()
            for _ in range(num_iter):
                model(dummy)
            elapsed = time_module.time() - start

        avg_ms = (elapsed / num_iter) * 1000

        results[model_name] = {
            **info,
            "inference_ms": round(avg_ms, 3),
            "throughput_fps": round(1000 / avg_ms, 1) if avg_ms > 0 else 0,
        }

        logger.info(f"{model_name}: {info['total_params']:,} params, "
                     f"{info['model_size_mb']:.1f} MB, {avg_ms:.2f} ms/img")

    if results:
        with open(output_dir / "efficiency.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Plots
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        params_data = {m: r["total_params"] / 1e6 for m, r in results.items()}
        plot_metric_comparison(
            params_data, "Parameters (M)",
            title="Model Parameter Count",
            save_path=str(plots_dir / "param_count.png"),
        )

        latency_data = {m: r["inference_ms"] for m, r in results.items()}
        plot_metric_comparison(
            latency_data, "Inference Time (ms)",
            title="Model Inference Latency",
            save_path=str(plots_dir / "inference_latency.png"),
        )

    return results


def main():
    args = parse_args()
    cfg = load_config(args.config, smoke_test=args.smoke_test)
    exp_config = Config.from_yaml(args.experiments_config)

    set_seed(cfg.get("project.seed", 42))
    logger = setup_logging("deepshield", cfg.get("logging.level", "INFO"),
                           log_dir=cfg.get("logging.log_dir"))
    device = get_device(args.device)

    experiments = exp_config.get("experiments", {})
    if hasattr(experiments, 'to_dict'):
        experiments = experiments.to_dict()

    target = args.experiment

    # Run experiments
    if target is None or target == "model_comparison":
        exp_cfg = experiments.get("model_comparison", {})
        if exp_cfg.get("enabled", True) or target:
            run_model_comparison(cfg, exp_cfg, device, args.smoke_test, logger)

    if target is None or target == "efficiency":
        exp_cfg = experiments.get("efficiency", {})
        if exp_cfg.get("enabled", True) or target:
            run_efficiency_experiment(cfg, exp_cfg, device, args.smoke_test, logger)

    logger.info("\n" + "=" * 60)
    logger.info("All experiments complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
