"""
DeepShield-X — Dataset Validation

Validate dataset structure, check image integrity, and report statistics.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)


# Expected directory structures for supported datasets
DATASET_STRUCTURES = {
    "faceforensics": {
        "required_dirs": [
            "original_sequences/youtube/c23/videos",
            "manipulated_sequences",
        ],
        "description": "FaceForensics++ dataset",
    },
    "celeb_df": {
        "required_dirs": [
            "Celeb-real",
            "Celeb-synthesis",
        ],
        "description": "Celeb-DF dataset",
    },
    "processed": {
        "required_dirs": [
            "faces",
        ],
        "description": "Pre-processed face dataset (after extraction)",
    },
}


def validate_dataset_structure(
    root: str,
    dataset_name: str = "faceforensics",
) -> Tuple[bool, List[str]]:
    """
    Validate that the dataset directory has the expected structure.

    Args:
        root: Path to the dataset root directory.
        dataset_name: Name of the dataset format to validate against.

    Returns:
        Tuple of (is_valid, list_of_issues).
    """
    root_path = Path(root)
    issues = []

    if not root_path.exists():
        issues.append(f"Dataset root does not exist: {root_path}")
        return False, issues

    if dataset_name not in DATASET_STRUCTURES:
        issues.append(f"Unknown dataset: {dataset_name}. Supported: {list(DATASET_STRUCTURES.keys())}")
        return False, issues

    structure = DATASET_STRUCTURES[dataset_name]

    for required_dir in structure["required_dirs"]:
        dir_path = root_path / required_dir
        if not dir_path.exists():
            issues.append(f"Missing required directory: {required_dir}")

    if issues:
        logger.warning(f"Dataset validation failed for '{dataset_name}': {len(issues)} issues")
        for issue in issues:
            logger.warning(f"  - {issue}")
        return False, issues

    logger.info(f"Dataset validation passed for '{dataset_name}' at {root_path}")
    return True, []


def validate_images(
    image_dir: str,
    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
    max_check: int = 1000,
) -> Dict[str, any]:
    """
    Validate image files in a directory.

    Checks that images can be opened, are RGB, and have valid dimensions.

    Args:
        image_dir: Directory containing images.
        extensions: Valid image file extensions.
        max_check: Maximum number of images to check.

    Returns:
        Dict with validation results and statistics.
    """
    image_path = Path(image_dir)
    if not image_path.exists():
        return {"valid": False, "error": f"Directory not found: {image_dir}"}

    all_files = []
    for ext in extensions:
        all_files.extend(image_path.rglob(f"*{ext}"))
        all_files.extend(image_path.rglob(f"*{ext.upper()}"))

    total = len(all_files)
    check_files = all_files[:max_check]
    valid_count = 0
    invalid_files = []
    sizes = []

    for fpath in check_files:
        try:
            img = Image.open(fpath)
            img.verify()
            img = Image.open(fpath)  # Re-open after verify
            if img.mode != "RGB":
                img = img.convert("RGB")
            sizes.append(img.size)
            valid_count += 1
        except Exception as e:
            invalid_files.append({"file": str(fpath), "error": str(e)})

    result = {
        "valid": len(invalid_files) == 0,
        "total_files": total,
        "checked": len(check_files),
        "valid_count": valid_count,
        "invalid_count": len(invalid_files),
        "invalid_files": invalid_files[:20],  # Cap at 20
    }

    if sizes:
        widths = [s[0] for s in sizes]
        heights = [s[1] for s in sizes]
        result["size_stats"] = {
            "min_width": min(widths),
            "max_width": max(widths),
            "min_height": min(heights),
            "max_height": max(heights),
            "common_size": max(set(sizes), key=sizes.count),
        }

    logger.info(
        f"Image validation: {valid_count}/{len(check_files)} valid "
        f"(total: {total}, checked: {len(check_files)})"
    )

    return result


def generate_metadata(
    faces_dir: str,
    output_path: str,
    dataset_name: str = "faceforensics",
) -> pd.DataFrame:
    """
    Generate metadata CSV from a directory of extracted faces.

    Expected structure:
        faces_dir/
            real/
                video_001/
                    frame_001.jpg
                    frame_002.jpg
            fake/
                video_002/
                    frame_001.jpg
                    frame_002.jpg

    Args:
        faces_dir: Directory with real/ and fake/ subdirectories.
        output_path: Path to save the metadata CSV.
        dataset_name: Name of the source dataset.

    Returns:
        DataFrame with columns: image_path, label, source_id, dataset.
    """
    faces_path = Path(faces_dir)
    records = []

    for label_name, label_int in [("real", 0), ("fake", 1)]:
        label_dir = faces_path / label_name
        if not label_dir.exists():
            logger.warning(f"Label directory not found: {label_dir}")
            continue

        for source_dir in sorted(label_dir.iterdir()):
            if not source_dir.is_dir():
                continue
            source_id = source_dir.name

            for img_file in sorted(source_dir.iterdir()):
                if img_file.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                    records.append({
                        "image_path": str(img_file.relative_to(faces_path)),
                        "label": label_int,
                        "label_name": label_name.upper(),
                        "source_id": source_id,
                        "dataset": dataset_name,
                    })

    df = pd.DataFrame(records)

    if len(df) > 0:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output, index=False)
        logger.info(
            f"Metadata generated: {len(df)} images "
            f"(REAL={sum(df['label']==0)}, FAKE={sum(df['label']==1)})"
        )
    else:
        logger.warning("No images found — metadata is empty")

    return df
