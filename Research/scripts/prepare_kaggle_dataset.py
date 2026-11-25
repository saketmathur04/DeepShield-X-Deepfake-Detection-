"""
DeepShield-X — Kaggle Dataset Adapter Script

Converts Kaggle 140k Real and Fake Faces (or similar Kaggle image datasets)
into DeepShield-X compatible metadata and splits.

Usage:
    python scripts/prepare_kaggle_dataset.py --input ./data/raw/real_vs_fake/real-vs-fake --output ./data
    # Generate a small test subset:
    python scripts/prepare_kaggle_dataset.py --input ./data/raw/real_vs_fake/real-vs-fake --output ./data --subset-size 100
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logging import setup_logging
from src.data.splitting import source_aware_split, check_leakage


def parse_args():
    parser = argparse.ArgumentParser(description="DeepShield-X Kaggle Dataset Adapter")
    parser.add_argument(
        "--input",
        type=str,
        default="./data/raw",
        help="Path to unzipped Kaggle dataset folder containing train/valid/test or real/fake",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data",
        help="Output directory for metadata and splits",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=None,
        help="Optional: number of samples PER CLASS PER SPLIT (e.g. 100) for testing",
    )
    return parser.parse_args()


def scan_directory(input_dir: Path) -> pd.DataFrame:
    """Scan directory structure for real and fake image files."""
    records = []
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}

    # Search for all image files recursively
    for path in input_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in image_extensions:
            parts = [p.lower() for p in path.parts]
            
            # Determine label
            if "real" in parts:
                label = 0
                label_name = "REAL"
            elif "fake" in parts:
                label = 1
                label_name = "FAKE"
            else:
                continue

            # Determine authoritative existing split hint if present
            split_hint = "unknown"
            for s in ["train", "valid", "val", "test"]:
                if s in parts:
                    split_hint = "val" if s == "valid" else s
                    break
            
            # Source ID: For Kaggle images, each image is a unique face (not video frames).
            # We use the filename/stem as the source_id to prevent any artificial leakage grouping.
            source_id = path.stem

            records.append({
                "image_path": str(path.resolve()),
                "label": label,
                "label_name": label_name,
                "source_id": source_id,
                "split_hint": split_hint,
                "dataset": "kaggle_140k",
            })

    return pd.DataFrame(records)


def create_subset(df: pd.DataFrame, subset_size: int, random_seed: int = 42) -> pd.DataFrame:
    """Create a balanced subset of the dataset."""
    rng = np.random.RandomState(random_seed)
    sampled = []
    
    # We want `subset_size` per class per split if splits exist
    for split_hint in df["split_hint"].unique():
        split_df = df[df["split_hint"] == split_hint]
        for label in [0, 1]:
            class_df = split_df[split_df["label"] == label]
            if len(class_df) > 0:
                n_samples = min(subset_size, len(class_df))
                indices = rng.choice(class_df.index, size=n_samples, replace=False)
                sampled.append(class_df.loc[indices])
                
    if not sampled:
        return df
        
    return pd.concat(sampled, ignore_index=True)


def main():
    args = parse_args()
    logger = setup_logging("deepshield")

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    logger.info("=" * 60)
    logger.info("DeepShield-X — Kaggle Dataset Preparation")
    logger.info(f"Input:  {input_dir}")
    logger.info(f"Output: {output_dir}")
    if args.subset_size:
        logger.info(f"Mode:   Subset (max {args.subset_size} samples per class/split)")
    logger.info("=" * 60)

    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    df = scan_directory(input_dir)
    if len(df) == 0:
        logger.error(f"No valid real/fake images found in {input_dir}")
        sys.exit(1)

    # Apply subset if requested
    if args.subset_size:
        df = create_subset(df, args.subset_size)

    logger.info(f"Found {len(df)} total images:")
    logger.info(f"  REAL: {sum(df['label'] == 0)}")
    logger.info(f"  FAKE: {sum(df['label'] == 1)}")

    # Ensure output dirs
    metadata_dir = output_dir / "metadata"
    splits_dir = output_dir / "splits"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    # Save full metadata
    df.to_csv(metadata_dir / "metadata.csv", index=False)
    logger.info(f"Saved full metadata to {metadata_dir / 'metadata.csv'}")

    # Process splits
    has_predefined_splits = set(df["split_hint"].unique()).issuperset({"train", "val", "test"})

    if has_predefined_splits:
        logger.info("Preserving existing authoritative train/valid/test splits...")
        train_df = df[df["split_hint"] == "train"].copy()
        val_df = df[df["split_hint"] == "val"].copy()
        test_df = df[df["split_hint"] == "test"].copy()
        
        # Verify no identical sources exist across splits
        check_leakage(train_df, val_df, test_df)
    else:
        logger.info("No authoritative splits found. Generating custom splits...")
        train_df, val_df, test_df = source_aware_split(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)

    # Save splits
    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir / "val.csv", index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)

    logger.info("Splits created/saved:")
    logger.info(f"  Train: {len(train_df)} samples")
    logger.info(f"  Val:   {len(val_df)} samples")
    logger.info(f"  Test:  {len(test_df)} samples")
    logger.info("=" * 60)
    logger.info("Kaggle dataset preparation complete!")


if __name__ == "__main__":
    main()
