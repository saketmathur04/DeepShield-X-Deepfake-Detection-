"""
DeepShield-X — Training Script

Usage:
    python scripts/train.py --model resnet50
    python scripts/train.py --model custom_cnn --smoke-test
    python scripts/train.py --config configs/default.yaml --model efficientnet
"""

import argparse
import json
import sys
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config, parse_cli_overrides
from src.utils.seed import set_seed, get_environment_info
from src.utils.device import get_device
from src.utils.logging import setup_logging
from src.models.factory import create_model, get_model_info
from src.data.dataset import SyntheticDataset, DeepFakeDataset, create_dataloader
from src.data.preprocessing import get_train_transforms, get_val_transforms
from src.training.trainer import Trainer
try:
    from src.training.losses import binary_pos_weight
except ImportError:
    def binary_pos_weight(num_negative: int, num_positive: int) -> float:
        if num_negative < 0 or num_positive <= 0:
            raise ValueError("Class counts must be non-negative and num_positive must be > 0")
        return float(num_negative / num_positive)
from src.utils.visualization import plot_training_curves
from src.evaluation.evaluator import Evaluator
from src.evaluation.metrics import compute_all_metrics, format_metrics_table
try:
    from src.evaluation.artifacts import write_training_history_plots
except ImportError:
    def write_training_history_plots(*args, **kwargs):
        return []


def parse_args():
    parser = argparse.ArgumentParser(description="DeepShield-X Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config YAML")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (overrides config)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run smoke test with synthetic data")
    parser.add_argument("--experiment-id", type=str, default="default",
                        help="Experiment identifier")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device (cuda/cpu/mps)")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip post-training evaluation on test set")
    parser.add_argument("--pos-weight", type=float, default=None,
                        help="Override positive class weight for BCEWithLogitsLoss")
    parser.add_argument("--auto-pos-weight", action="store_true",
                        help="Balance BCE labels 0=REAL and 1=FAKE with pos_weight = REAL / FAKE")
    parser.add_argument("--monitor", type=str, default=None,
                        help="Metric to monitor for checkpoint selection and early stopping (e.g., val_roc_auc, val_loss)")
    parser.add_argument("--enable-early-stopping", action="store_true",
                        help="Enable early stopping")
    parser.add_argument("--patience", type=int, default=None,
                        help="Override early stopping patience")
    parser.add_argument("--min-delta", type=float, default=None,
                        help="Override early stopping min delta")
    parser.add_argument("--enable-augmentation", action="store_true",
                        help="Enable moderate training data augmentation")
    return parser.parse_args()


def print_baseline_diagnostics(logger, cfg, train_dataset, val_dataset):
    """FIX 10: Print baseline diagnostics before training."""
    logger.info("=" * 60)
    logger.info("BASELINE DIAGNOSTICS")
    logger.info("=" * 60)

    # Class distribution
    for name, ds in [("TRAIN", train_dataset), ("VAL", val_dataset)]:
        total = len(ds)
        if hasattr(ds, 'num_real'):
            real_count = ds.num_real
            fake_count = ds.num_fake
        else:
            real_count = fake_count = 0
        logger.info(f"{name}: REAL={real_count} ({100*real_count/total:.1f}%)  "
                     f"FAKE={fake_count} ({100*fake_count/total:.1f}%)  "
                     f"Total={total}")

    # Label mapping
    logger.info("Label mapping: 0=REAL, 1=FAKE")

    # Key config values
    training_cfg = cfg.get("training", {})
    pos_weight_val = training_cfg.get("pos_weight", None) if hasattr(training_cfg, 'get') else None
    aug_cfg = cfg.get("augmentation", {})
    aug_enabled = aug_cfg.get("enabled", True) if hasattr(aug_cfg, 'get') else True
    es_cfg = training_cfg.get("early_stopping", {}) if hasattr(training_cfg, 'get') else {}
    es_enabled = es_cfg.get("enabled", True) if hasattr(es_cfg, 'get') else True
    monitor_val = training_cfg.get("checkpoint_monitor") or es_cfg.get("monitor", "val_loss")

    logger.info(f"pos_weight: {pos_weight_val}")
    logger.info(f"augmentation: {'enabled' if aug_enabled else 'DISABLED'}")
    logger.info(f"checkpoint_monitor: {monitor_val}")
    logger.info(f"early_stopping: {'enabled' if es_enabled else 'DISABLED'} (patience={es_cfg.get('patience', 5)}, min_delta={es_cfg.get('min_delta', 0.001)})")
    logger.info(f"optimizer: {training_cfg.get('optimizer', 'adamw') if hasattr(training_cfg, 'get') else 'adamw'}")
    logger.info(f"scheduler: {training_cfg.get('scheduler', 'cosine') if hasattr(training_cfg, 'get') else 'cosine'}")
    logger.info(f"learning_rate: {training_cfg.get('learning_rate', 1e-4) if hasattr(training_cfg, 'get') else 1e-4}")
    logger.info(f"resume_from: {training_cfg.get('resume_from', None) if hasattr(training_cfg, 'get') else None}")
    logger.info("=" * 60)


def run_sanity_check(model, train_loader, device, logger):
    """FIX 11: Raw logit sanity check before training."""
    logger.info("=" * 60)
    logger.info("SANITY CHECK — One-batch forward pass")
    logger.info("=" * 60)

    model.to(device)
    model.eval()
    with torch.no_grad():
        images, labels, _ = next(iter(train_loader))
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images).squeeze(-1)
        probs = torch.sigmoid(logits)

        logger.info(f"Batch size: {images.shape[0]}")
        logger.info(f"Logit  — min={logits.min().item():.4f}  max={logits.max().item():.4f}  mean={logits.mean().item():.4f}")
        logger.info(f"Prob   — min={probs.min().item():.4f}  max={probs.max().item():.4f}  mean={probs.mean().item():.4f}")
        logger.info(f"Labels — REAL={int((labels == 0).sum())}  FAKE={int((labels == 1).sum())}")

    model.train()
    logger.info("Sanity check PASSED")
    logger.info("=" * 60)


def main():
    args = parse_args()

    # Load config
    cfg = load_config(args.config, smoke_test=args.smoke_test)

    # Apply CLI overrides
    overrides = {}
    if args.model:
        overrides.setdefault("model", {})["name"] = args.model
    if args.epochs:
        overrides.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size:
        overrides.setdefault("training", {})["batch_size"] = args.batch_size
    if args.lr:
        overrides.setdefault("training", {})["learning_rate"] = args.lr
    if args.pos_weight is not None:
        overrides.setdefault("training", {})["pos_weight"] = args.pos_weight
    if args.monitor:
        overrides.setdefault("training", {}).setdefault("early_stopping", {})["monitor"] = args.monitor
        overrides["training"]["checkpoint_monitor"] = args.monitor
    if args.enable_early_stopping:
        overrides.setdefault("training", {}).setdefault("early_stopping", {})["enabled"] = True
    if args.patience is not None:
        overrides.setdefault("training", {}).setdefault("early_stopping", {})["patience"] = args.patience
    if args.min_delta is not None:
        overrides.setdefault("training", {}).setdefault("early_stopping", {})["min_delta"] = args.min_delta
    if args.enable_augmentation:
        overrides.setdefault("augmentation", {})["enabled"] = True
    if overrides:
        cfg.merge(overrides)

    # Setup
    seed = cfg.get("project.seed", 42)
    set_seed(seed)

    logger = setup_logging(
        name="deepshield",
        level=cfg.get("logging.level", "INFO"),
        log_dir=cfg.get("logging.log_dir", "./experiments/logs"),
    )

    device = get_device(args.device)
    model_name = cfg.get("model.name", "resnet50")

    logger.info("=" * 60)
    logger.info("DeepShield-X Training")
    logger.info("=" * 60)
    logger.info(f"Model: {model_name}")
    logger.info(f"Device: {device}")
    logger.info(f"Smoke test: {args.smoke_test}")
    logger.info(f"Seed: {seed}")

    # Save environment info
    env_info = get_environment_info()
    output_dir = Path(cfg.get("project.output_dir", "./experiments"))
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "environment.json", "w") as f:
        json.dump(env_info, f, indent=2)

    # Create model
    model = create_model(
        name=model_name,
        num_classes=cfg.get("model.num_classes", 1),
        pretrained=cfg.get("model.pretrained", True),
        dropout=cfg.get("model.dropout", 0.5),
        finetune_strategy=cfg.get("model.finetune_strategy", "full"),
    ).to(device)

    model_info = get_model_info(model, model_name)
    logger.info(f"Model params: {model_info['total_params']:,} "
                f"(trainable: {model_info['trainable_params']:,})")

    # Create datasets
    image_size = cfg.get("dataset.image_size", 224)
    batch_size = cfg.get("training.batch_size", 32)
    num_workers = cfg.get("dataset.num_workers", 4)

    # Variables for test eval (scoped here so they're available later)
    root_dir = None
    val_transform = None
    splits_dir = None

    if args.smoke_test:
        # Use synthetic data for smoke test
        smoke_cfg = cfg.get("smoke_test", {})
        num_samples = smoke_cfg.num_samples if hasattr(smoke_cfg, 'num_samples') else 50
        image_size = smoke_cfg.image_size if hasattr(smoke_cfg, 'image_size') else 64

        train_dataset = SyntheticDataset(num_samples=num_samples, image_size=image_size, seed=seed)
        val_dataset = SyntheticDataset(num_samples=max(num_samples // 4, 10), image_size=image_size, seed=seed + 1)

        train_loader = create_dataloader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        val_loader = create_dataloader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
    else:
        # Use real dataset
        aug_config = cfg.get("augmentation", {})
        aug_dict = aug_config.to_dict() if hasattr(aug_config, 'to_dict') else aug_config

        train_transform = get_train_transforms(image_size, aug_dict)
        val_transform = get_val_transforms(image_size)

        splits_dir = Path(cfg.get("dataset.root", "./data")) / "splits"
        train_csv = splits_dir / "train.csv"
        val_csv = splits_dir / "val.csv"

        if not train_csv.exists():
            logger.error(f"Training split not found: {train_csv}")
            logger.error("Run: python scripts/prepare_dataset.py first")
            sys.exit(1)

        # FIX 6: root_dir = dataset root (NOT appending /faces)
        # CSV paths already start with faces/... so root resolves correctly
        root_dir = str(Path(cfg.get("dataset.root", "./data")))
        logger.info(f"Dataset root: {root_dir}")

        train_dataset = DeepFakeDataset(str(train_csv), root_dir, train_transform, image_size)
        val_dataset = DeepFakeDataset(str(val_csv), root_dir, val_transform, image_size)

        train_loader = create_dataloader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader = create_dataloader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")

    # BCEWithLogitsLoss's positive class is label 1 (FAKE), so the balanced
    # weight is REAL / FAKE.  This intentionally is < 1 when FAKE is majority.
    if getattr(args, "auto_pos_weight", False):
        if hasattr(train_dataset, "num_real") and hasattr(train_dataset, "num_fake") and train_dataset.num_fake > 0:
            auto_weight = binary_pos_weight(train_dataset.num_real, train_dataset.num_fake)
            cfg.merge({"training": {"pos_weight": auto_weight}})
            logger.info(
                "Auto-computed BCE pos_weight for positive label 1=FAKE: "
                f"{train_dataset.num_real} REAL / {train_dataset.num_fake} FAKE = {auto_weight:.6f}"
            )

    # FIX 10: Baseline diagnostics
    print_baseline_diagnostics(logger, cfg, train_dataset, val_dataset)

    # FIX 11: Sanity check — one-batch forward pass
    run_sanity_check(model, train_loader, device, logger)

    # Create trainer
    trainer = Trainer(
        model=model,
        device=device,
        config=cfg.to_dict(),
    )

    # Train
    result = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        model_name=model_name,
        experiment_id=args.experiment_id,
    )

    # Future experiments receive per-metric, publication-ready training plots.
    # This only writes new files for the experiment being trained.
    exp_dir_name = args.experiment_id if args.experiment_id != "default" else model_name
    experiment_results_dir = Path(cfg.get("evaluation.results_dir", "./experiments/results")) / exp_dir_name
    write_training_history_plots(result["history"], str(experiment_results_dir), model_name)

    # Plot training curves
    plots_dir = Path(cfg.get("evaluation.plots_dir", "./experiments/plots"))
    plot_filename = f"{args.experiment_id}_{model_name}_training_curves.png" if args.experiment_id != "default" else f"{model_name}_training_curves.png"
    plot_training_curves(
        result["history"],
        model_name=model_name,
        save_path=str(plots_dir / plot_filename),
    )

    # ---- FIX 8/9: Post-training evaluation on test set ----
    if not args.smoke_test and not args.skip_eval and splits_dir is not None:
        test_csv = splits_dir / "test.csv"
        if test_csv.exists():
            logger.info("=" * 60)
            logger.info("Post-Training Evaluation on Test Set")
            logger.info("=" * 60)

            test_dataset = DeepFakeDataset(
                str(test_csv), root_dir, val_transform, image_size
            )
            test_loader = create_dataloader(
                test_dataset,
                batch_size=cfg.get("evaluation.batch_size", 64),
                shuffle=False,
                num_workers=num_workers,
            )
            logger.info(f"Test samples: {len(test_dataset)}")

            # Test set class distribution
            logger.info(f"TEST: REAL={test_dataset.num_real}  FAKE={test_dataset.num_fake}")

            # Load best checkpoint
            best_ckpt = result["best_checkpoint"]
            logger.info(f"Loading best checkpoint: {best_ckpt}")
            checkpoint = torch.load(best_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])

            # Run evaluation
            evaluator = Evaluator(
                model=model,
                device=device,
                threshold=cfg.get("evaluation.threshold", 0.5),
                use_amp=cfg.get("training.mixed_precision", True),
            )
            predictions = evaluator.predict(test_loader)

            # Compute all metrics
            metrics = compute_all_metrics(
                y_true=predictions["y_true"],
                y_pred=predictions["y_pred"],
                y_prob=predictions["y_prob"],
                threshold=cfg.get("evaluation.threshold", 0.5),
            )

            # Save metrics — isolated per experiment_id
            results_dir = experiment_results_dir
            results_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = results_dir / f"{model_name}_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Metrics saved to: {metrics_path}")

            # Write full evaluation package (curves, confusion matrix, errors, calibration)
            try:
                from src.evaluation.artifacts import write_evaluation_package
                eval_pkg_dir = results_dir / "evaluation" / "test"
                write_evaluation_package(
                    y_true=predictions["y_true"],
                    y_prob=predictions["y_prob"],
                    metadata=predictions["metadata"],
                    output_dir=str(eval_pkg_dir),
                    experiment_id=args.experiment_id,
                    model_name=model_name,
                    split="test",
                    threshold=cfg.get("evaluation.threshold", 0.5),
                    config=cfg.to_dict(),
                    checkpoint_path=str(best_ckpt),
                    splits_dir=splits_dir,
                    dataset_name="FaceForensics++ C23",
                )
                logger.info(f"Full evaluation package written to: {eval_pkg_dir}")
            except Exception as e:
                logger.warning(f"Could not write full evaluation package: {e}")

            # Log key metrics
            logger.info(f"Test Accuracy:   {metrics['accuracy']:.4f}")
            logger.info(f"Test Precision:  {metrics['precision']:.4f}")
            logger.info(f"Test Recall:     {metrics['recall']:.4f}")
            logger.info(f"Test F1:         {metrics['f1']:.4f}")
            logger.info(f"Test ROC-AUC:    {metrics.get('roc_auc', 'N/A')}")
            logger.info(f"Test PR-AUC:     {metrics.get('pr_auc', 'N/A')}")
            logger.info(f"Specificity:     {metrics['specificity']:.4f}")
            logger.info(f"Confusion Matrix: TP={metrics['true_positives']} "
                        f"TN={metrics['true_negatives']} "
                        f"FP={metrics['false_positives']} "
                        f"FN={metrics['false_negatives']}")
        else:
            logger.warning(f"Test split not found: {test_csv} — skipping evaluation")

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"Best checkpoint: {result['best_checkpoint']}")
    logger.info(f"Final val_acc: {result['final_val_acc']:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
