"""
DeepShield-X — Create Splits Script

Usage:
    python scripts/create_splits.py --metadata ./data/metadata/metadata.csv --output ./data/splits
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.utils.logging import setup_logging
from src.utils.seed import set_seed
from src.data.splitting import source_aware_split, save_splits


def parse_args():
    parser = argparse.ArgumentParser(description="DeepShield-X Create Splits")
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--output", type=str, default="./data/splits")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging("deepshield")
    set_seed(args.seed)

    df = pd.read_csv(args.metadata)
    logger.info(f"Loaded metadata: {len(df)} samples")

    train_df, val_df, test_df = source_aware_split(
        df, args.train_ratio, args.val_ratio, args.test_ratio, args.seed
    )
    save_splits(train_df, val_df, test_df, args.output)

    logger.info("Splits created successfully")


if __name__ == "__main__":
    main()
