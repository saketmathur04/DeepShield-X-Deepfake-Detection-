"""
DeepShield-X — Dataset Preparation Script

Usage:
    python scripts/prepare_dataset.py --dataset-root ./data/raw/faceforensics
    python scripts/prepare_dataset.py --smoke-test
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logging import setup_logging
from src.data.validation import validate_dataset_structure, validate_images, generate_metadata
from src.data.splitting import source_aware_split, save_splits
from src.data.dataset import SyntheticDataset


def parse_args():
    parser = argparse.ArgumentParser(description="DeepShield-X Dataset Preparation")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--dataset-name", type=str, default="faceforensics")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Generate synthetic dataset for testing")
    parser.add_argument("--output-dir", type=str, default="./data")
    return parser.parse_args()


def prepare_smoke_test(output_dir: str, seed: int = 42):
    """Generate synthetic data for smoke testing."""
    logger = setup_logging("deepshield")
    logger.info("Generating synthetic smoke-test dataset...")

    import numpy as np
    import os

    faces_dir = Path(output_dir) / "faces"
    metadata_dir = Path(output_dir) / "metadata"
    splits_dir = Path(output_dir) / "splits"

    # Create synthetic face images
    rng = np.random.RandomState(seed)
    records = []

    for label_name, label_int in [("real", 0), ("fake", 1)]:
        for video_id in range(5):
            source_id = f"{label_name}_video_{video_id:03d}"
            source_dir = faces_dir / label_name / source_id
            source_dir.mkdir(parents=True, exist_ok=True)

            for frame_id in range(10):
                img = rng.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                if label_int == 1:
                    # Add subtle pattern to fake images
                    img[::4, :, :] = img[::4, :, :] // 2

                fname = f"frame_{frame_id:04d}.jpg"
                fpath = source_dir / fname

                from PIL import Image
                Image.fromarray(img).save(str(fpath))

                records.append({
                    "image_path": f"{label_name}/{source_id}/{fname}",
                    "label": label_int,
                    "label_name": label_name.upper(),
                    "source_id": source_id,
                    "dataset": "synthetic",
                })

    # Save metadata
    import pandas as pd
    metadata_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(metadata_dir / "metadata.csv", index=False)

    # Create splits
    train_df, val_df, test_df = source_aware_split(
        df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=seed
    )
    save_splits(train_df, val_df, test_df, str(splits_dir))

    logger.info(f"Smoke-test dataset created at: {output_dir}")
    logger.info(f"  Total: {len(df)} images ({df['label'].value_counts().to_dict()})")
    logger.info(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("project.seed", 42))
    logger = setup_logging("deepshield")

    if args.smoke_test:
        prepare_smoke_test(args.output_dir, seed=cfg.get("project.seed", 42))
        return

    dataset_root = args.dataset_root or cfg.get("dataset.root", "./data/raw")

    logger.info("=" * 60)
    logger.info("DeepShield-X Dataset Preparation")
    logger.info("=" * 60)

    # Step 1: Validate structure
    logger.info("Step 1: Validating dataset structure...")
    is_valid, issues = validate_dataset_structure(dataset_root, args.dataset_name)
    if not is_valid:
        logger.error("Dataset validation failed. Please check the dataset.")
        for issue in issues:
            logger.error(f"  - {issue}")
        sys.exit(1)

    # Step 2: Validate images
    faces_dir = Path(args.output_dir) / "faces"
    if faces_dir.exists():
        logger.info("Step 2: Validating images...")
        img_report = validate_images(str(faces_dir))
        logger.info(f"  Valid: {img_report['valid_count']}/{img_report['checked']}")

    # Step 3: Generate metadata
    logger.info("Step 3: Generating metadata...")
    metadata_path = Path(args.output_dir) / "metadata" / "metadata.csv"
    df = generate_metadata(str(faces_dir), str(metadata_path), args.dataset_name)

    if len(df) == 0:
        logger.error("No images found. Run face extraction first.")
        sys.exit(1)

    # Step 4: Create splits
    logger.info("Step 4: Creating source-aware splits...")
    split_ratios = cfg.get("dataset.split_ratios", {})
    train_ratio = split_ratios.train if hasattr(split_ratios, 'train') else 0.7
    val_ratio = split_ratios.val if hasattr(split_ratios, 'val') else 0.15
    test_ratio = split_ratios.test if hasattr(split_ratios, 'test') else 0.15

    train_df, val_df, test_df = source_aware_split(
        df, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
        seed=cfg.get("project.seed", 42)
    )

    splits_dir = Path(args.output_dir) / "splits"
    save_splits(train_df, val_df, test_df, str(splits_dir))

    logger.info("=" * 60)
    logger.info("Dataset preparation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
