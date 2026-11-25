"""
DeepShield-X — Evaluation Script

Usage:
    python scripts/evaluate.py --model resnet50 --checkpoint experiments/checkpoints/default/resnet50_best.pth
    python scripts/evaluate.py --model custom_cnn --smoke-test
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.device import get_device
from src.utils.logging import setup_logging
from src.models.factory import create_model
from src.data.dataset import SyntheticDataset, DeepFakeDataset, create_dataloader
from src.data.preprocessing import get_val_transforms
from src.evaluation.evaluator import Evaluator
from src.evaluation.confusion_matrix import plot_confusion_matrix
from src.evaluation.roc import plot_roc_curve, plot_pr_curve
from src.utils.visualization import plot_confidence_distribution
from src.utils.checkpoint import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="DeepShield-X Evaluation")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, smoke_test=args.smoke_test)

    set_seed(cfg.get("project.seed", 42))
    logger = setup_logging("deepshield", cfg.get("logging.level", "INFO"))
    device = get_device(args.device)

    model_name = args.model
    output_dir = args.output_dir or str(
        Path(cfg.get("evaluation.results_dir", "./experiments/results")) / model_name
    )
    plots_dir = str(Path(cfg.get("evaluation.plots_dir", "./experiments/plots")) / model_name)

    logger.info("=" * 60)
    logger.info(f"DeepShield-X Evaluation — {model_name}")
    logger.info("=" * 60)

    # Create model
    model = create_model(model_name, pretrained=False if args.checkpoint else True)

    # Load checkpoint if provided
    if args.checkpoint:
        ckpt_info = load_checkpoint(args.checkpoint, model, device=device)
        logger.info(f"Loaded checkpoint: epoch={ckpt_info['epoch']}")

    # Create test dataset
    image_size = cfg.get("dataset.image_size", 224)
    batch_size = cfg.get("evaluation.batch_size", 64)

    if args.smoke_test:
        test_dataset = SyntheticDataset(
            num_samples=30, image_size=image_size, seed=99
        )
        test_loader = create_dataloader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
    else:
        val_transform = get_val_transforms(image_size)
        splits_dir = Path(cfg.get("dataset.root", "./data")) / "splits"
        test_csv = splits_dir / "test.csv"

        if not test_csv.exists():
            logger.error(f"Test split not found: {test_csv}")
            sys.exit(1)

        root_dir = str(Path(cfg.get("dataset.root", "./data")) / "faces")
        test_dataset = DeepFakeDataset(str(test_csv), root_dir, val_transform, image_size)
        test_loader = create_dataloader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=cfg.get("dataset.num_workers", 4)
        )

    # Evaluate
    evaluator = Evaluator(model, device, threshold=cfg.get("evaluation.threshold", 0.5))
    result = evaluator.evaluate(test_loader, model_name=model_name, output_dir=output_dir)

    # Generate plots
    predictions = result["predictions"]

    plot_confusion_matrix(
        predictions["y_true"], predictions["y_pred"],
        title=f"Confusion Matrix — {model_name}",
        save_path=f"{plots_dir}/confusion_matrix.png",
    )

    plot_roc_curve(
        fpr=result["curves"]["roc"]["fpr"],
        tpr=result["curves"]["roc"]["tpr"],
        roc_auc=result["metrics"]["roc_auc"],
        model_name=model_name,
        save_path=f"{plots_dir}/roc_curve.png",
    )

    plot_pr_curve(
        precision=result["curves"]["pr"]["precision"],
        recall=result["curves"]["pr"]["recall"],
        pr_auc=result["metrics"]["pr_auc"],
        model_name=model_name,
        save_path=f"{plots_dir}/pr_curve.png",
    )

    plot_confidence_distribution(
        predictions["y_true"], predictions["y_prob"],
        save_path=f"{plots_dir}/confidence_dist.png",
    )

    logger.info("=" * 60)
    logger.info("Evaluation complete!")
    logger.info(f"Results: {output_dir}")
    logger.info(f"Plots: {plots_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
