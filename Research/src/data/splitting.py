"""
DeepShield-X — Source-Level Data Splitting

Splits data at the VIDEO/SOURCE level to prevent data leakage.
Frames from the same source video never appear in different splits.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class UnionFind:
    """Union-Find data structure for grouping connected identities."""
    def __init__(self):
        self.parent = {}
        
    def find(self, i: str) -> str:
        if i not in self.parent:
            self.parent[i] = i
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
        
    def union(self, i: str, j: str) -> None:
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j
            
    def get_groups(self) -> Dict[str, List[str]]:
        groups = {}
        for i in self.parent:
            root = self.find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)
        return groups


def build_identity_groups(metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Build connected components of identities using Union-Find.
    
    Expects metadata to have either an 'identities' column (list of strings)
    or a 'source_id' column (string).
    """
    uf = UnionFind()
    
    # Extract identities
    if "identities" in metadata.columns:
        identity_lists = metadata["identities"]
    elif "source_id" in metadata.columns:
        identity_lists = metadata["source_id"].apply(lambda x: [x])
    else:
        raise ValueError("Metadata must contain 'identities' or 'source_id'")
        
    # Build components
    for ids in identity_lists:
        if not isinstance(ids, list):
            ids = [ids]
        if len(ids) == 0:
            continue
        first_id = str(ids[0])
        uf.find(first_id)
        for other_id in ids[1:]:
            uf.union(first_id, str(other_id))
            
    # Map each row to its root group
    def get_root_group(ids):
        if not isinstance(ids, list):
            ids = [ids]
        if len(ids) == 0:
            return "unknown"
        return uf.find(str(ids[0]))
        
    df = metadata.copy()
    df["group_id"] = identity_lists.apply(get_root_group)
    return df



def source_aware_split(
    metadata: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    stratify_by_label: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split metadata into train/val/test at the SOURCE level.

    All frames from a given source_id go to the SAME split.
    This prevents data leakage between splits.

    Args:
        metadata: DataFrame with 'source_id' and 'label' columns.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for testing.
        seed: Random seed.
        stratify_by_label: Maintain label distribution across splits.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"

    if "group_id" not in metadata.columns:
        metadata = build_identity_groups(metadata)

    rng = np.random.RandomState(seed)

    if stratify_by_label:
        train_groups, val_groups, test_groups = [], [], []
        
        # Determine a single primary label for each group to prevent double-assignment
        # across different labels (e.g. when a group has both REAL and FAKE videos).
        group_primary_labels = metadata.drop_duplicates("group_id").set_index("group_id")["label"]

        for label in group_primary_labels.unique():
            label_groups = group_primary_labels[group_primary_labels == label].index.to_numpy()
            rng.shuffle(label_groups)

            n = len(label_groups)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)

            train_groups.extend(label_groups[:n_train])
            val_groups.extend(label_groups[n_train:n_train + n_val])
            test_groups.extend(label_groups[n_train + n_val:])
    else:
        all_groups = metadata["group_id"].unique()
        rng.shuffle(all_groups)

        n = len(all_groups)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_groups = all_groups[:n_train]
        val_groups = all_groups[n_train:n_train + n_val]
        test_groups = all_groups[n_train + n_val:]

    train_df = metadata[metadata["group_id"].isin(train_groups)].copy()
    val_df = metadata[metadata["group_id"].isin(val_groups)].copy()
    test_df = metadata[metadata["group_id"].isin(test_groups)].copy()

    # Add split column
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    logger.info(
        f"Source-aware split: "
        f"train={len(train_df)} ({len(set(train_groups))} groups), "
        f"val={len(val_df)} ({len(set(val_groups))} groups), "
        f"test={len(test_df)} ({len(set(test_groups))} groups)"
    )

    # Run leakage check
    check_leakage(train_df, val_df, test_df)

    return train_df, val_df, test_df


def check_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """
    Verify no source_id appears in multiple splits. Raises ValueError on leakage.
    """
    def get_all_identities(df):
        if "identities" in df.columns:
            all_ids = set()
            for ids in df["identities"]:
                if isinstance(ids, list):
                    all_ids.update([str(i) for i in ids])
                else:
                    all_ids.add(str(ids))
            return all_ids
        elif "source_id" in df.columns:
            return set(df["source_id"].astype(str).unique())
        return set()

    train_sources = get_all_identities(train_df)
    val_sources = get_all_identities(val_df)
    test_sources = get_all_identities(test_df)

    train_val_overlap = train_sources & val_sources
    train_test_overlap = train_sources & test_sources
    val_test_overlap = val_sources & test_sources

    leaks = []
    if train_val_overlap:
        leaks.append(f"Train-Val overlap: {train_val_overlap}")
    if train_test_overlap:
        leaks.append(f"Train-Test overlap: {train_test_overlap}")
    if val_test_overlap:
        leaks.append(f"Val-Test overlap: {val_test_overlap}")

    if leaks:
        error_msg = "DATA LEAKAGE DETECTED!\n" + "\n".join(leaks)
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info("Leakage check passed — no source overlap between splits")


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str,
) -> Dict[str, str]:
    """
    Save split DataFrames as CSVs.

    Args:
        train_df, val_df, test_df: Split DataFrames.
        output_dir: Directory to save split files.

    Returns:
        Dict mapping split name to file path.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    paths = {}
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        fpath = out_path / f"{name}.csv"
        df.to_csv(fpath, index=False)
        paths[name] = str(fpath)
        logger.info(f"Saved {name} split: {len(df)} samples → {fpath}")

    # Save combined metadata with splits
    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
    combined_path = out_path / "all_splits.csv"
    combined.to_csv(combined_path, index=False)
    paths["all"] = str(combined_path)

    # Save split summary
    summary = {
        "train": {"samples": len(train_df), "sources": train_df["source_id"].nunique()},
        "val": {"samples": len(val_df), "sources": val_df["source_id"].nunique()},
        "test": {"samples": len(test_df), "sources": test_df["source_id"].nunique()},
    }
    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if "label" in df.columns:
            summary[split_name]["real"] = int((df["label"] == 0).sum())
            summary[split_name]["fake"] = int((df["label"] == 1).sum())

    import json
    summary_path = out_path / "split_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return paths
