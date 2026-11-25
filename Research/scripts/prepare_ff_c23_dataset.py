"""
DeepShield-X — FaceForensics++ C23 Dataset Adapter Script

Prepares the FF++ C23 video dataset (Kaggle version).
Uses Union-Find connected components to prevent identity leakage across splits.
Splits videos BEFORE face extraction to prevent storage duplication.

Usage:
    python scripts/prepare_ff_c23_dataset.py --input ./data/raw/ff-c23 --output ./data --max-videos 50 --frames-per-video 5
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logging import setup_logging
from src.data.splitting import source_aware_split, check_leakage, build_identity_groups
from src.data.face_extraction import FaceExtractor


def parse_args():
    parser = argparse.ArgumentParser(description="DeepShield-X FF++ C23 Preparation")
    parser.add_argument("--input", type=str, required=True, help="Path to unzipped FF++ C23 directory")
    parser.add_argument("--output", type=str, default="./data", help="Output directory")
    parser.add_argument("--max-videos", type=int, default=None, help="Maximum number of videos to process (for small tests)")
    parser.add_argument("--frames-per-video", type=int, default=10, help="Number of frames to extract per video")
    parser.add_argument("--image-size", type=int, default=224, help="Extracted face size")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    return parser.parse_args()


def get_identities(filename: str, category: str) -> list:
    """
    Extract base identities from FF++ filename.
    Originals: '001.mp4' -> ['001']
    Manipulated: '001_002.mp4' -> ['001', '002']
    Returns empty list if unresolved.
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    
    if category == "original":
        if len(parts) == 1 and parts[0].isdigit():
            return [parts[0]]
        # YouTube originals often just have 3 digits
        elif len(parts) == 1:
            return [parts[0]]
            
    # DeepFakeDetection uses different IDs (e.g., actors), usually long strings
    # Real data pattern: 01_02__meeting_serious__YVGY8LOK.mp4 -> source=01, target=02
    if category == "DeepFakeDetection":
        if "__" in stem:
            # e.g. 01_02__meeting_serious__YVGY8LOK
            prefix = stem.split("__")[0] # '01_02' or '01'
            prefix_parts = prefix.split("_")
            if len(prefix_parts) >= 2:
                return [prefix_parts[0], prefix_parts[1]]
            elif len(prefix_parts) == 1:
                return [prefix_parts[0]]
        
        # Fallback to older patterns if present
        if len(parts) == 3:
            return parts # e.g. actor1_actor2_123
        return [stem]

    # For manipulations like Face2Face, Deepfakes: typically 001_002
    if len(parts) == 2:
        return [parts[0], parts[1]]
        
    return [] # Unresolved


def scan_ff_videos(input_dir: Path) -> tuple:
    """
    Scan raw FF++ directory and identify original vs manipulated videos.
    Returns (resolved_df, unresolved_df)
    """
    records = []
    unresolved = []
    video_exts = {".mp4", ".avi"}
    
    for path in input_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in video_exts:
            parts_lower = [p.lower() for p in path.parts]
            
            # Determine category
            category = "unknown"
            label = 1
            label_name = "FAKE"
            
            if "original" in parts_lower or "original_sequences" in parts_lower or "youtube" in parts_lower:
                category = "original"
                label = 0
                label_name = "REAL"
            elif "deepfakes" in parts_lower: category = "Deepfakes"
            elif "face2face" in parts_lower: category = "Face2Face"
            elif "faceswap" in parts_lower: category = "FaceSwap"
            elif "faceshifter" in parts_lower: category = "FaceShifter"
            elif "neuraltextures" in parts_lower: category = "NeuralTextures"
            elif "deepfakedetection" in parts_lower: category = "DeepFakeDetection"
            else:
                unresolved.append({"video_path": str(path), "manipulation": category, "reason": "Unknown category format"})
                continue
                    
            identities = get_identities(path.name, category)
            
            if not identities:
                unresolved.append({"video_path": str(path), "manipulation": category, "reason": "Could not parse identity from filename"})
                continue
            
            records.append({
                "video_path": str(path.resolve()),
                "category": category,
                "label": label,
                "label_name": label_name,
                "identities": identities,
                "video_name": path.stem,
            })
            
    return pd.DataFrame(records), pd.DataFrame(unresolved)


def extract_faces_for_split(
    split_df: pd.DataFrame, 
    split_name: str, 
    output_dir: Path, 
    extractor: FaceExtractor, 
    frames_per_video: int
) -> pd.DataFrame:
    """Extract frames for a specific split and generate metadata records."""
    faces_dir = output_dir / "faces" / split_name
    records = []
    logger = logging.getLogger(__name__)
    
    logger.info(f"Extracting {len(split_df)} videos for {split_name} split...")
    
    for idx, row in split_df.iterrows():
        vid_path = row["video_path"]
        label_name = row["label_name"].lower()
        
        # Create output directory
        out_sub_dir = faces_dir / label_name / row["category"] / row["video_name"]
        out_sub_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            logger.warning(f"Failed to open {vid_path}")
            continue
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            continue
            
        step = max(1, total_frames // frames_per_video)
        indices_to_extract = [i * step for i in range(frames_per_video)]
        indices_to_extract = [i for i in indices_to_extract if i < total_frames]
        
        for frame_idx in indices_to_extract:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret: continue
                
            faces_bbox = extractor.detect_faces(frame)
            if not faces_bbox: continue
                
            largest_bbox = max(faces_bbox, key=lambda b: b[2] * b[3])
            face_img = extractor.crop_face(frame, largest_bbox)
            
            face_filename = f"frame_{frame_idx:04d}.jpg"
            face_path = out_sub_dir / face_filename
            cv2.imwrite(str(face_path), face_img)
            
            rel_path = face_path.relative_to(output_dir)
            records.append({
                "image_path": str(rel_path).replace("\\", "/"),
                "label": row["label"],
                "label_name": row["label_name"],
                "manipulation": row["category"],
                "identities": row["identities"],
                "video_id": row["video_name"],
                "frame_id": frame_idx,
                "split": split_name,
            })
            
        cap.release()
        
    return pd.DataFrame(records)


def main():
    args = parse_args()
    logger = setup_logging("deepshield")
    
    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    
    logger.info("=" * 60)
    logger.info("DeepShield-X — FF++ C23 Dataset Preparation (Union-Find Leakage Protection)")
    logger.info(f"Input: {input_dir}")
    if args.max_videos:
        logger.info(f"LIMIT ACTIVE: max {args.max_videos} videos (Small Test Mode)")
    logger.info("=" * 60)
    
    if not input_dir.exists():
        logger.error(f"Directory not found: {input_dir}")
        sys.exit(1)
        
    # 1. Scan Videos
    logger.info("Scanning video directory...")
    df_videos, df_unresolved = scan_ff_videos(input_dir)
    
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    if len(df_unresolved) > 0:
        logger.warning(f"Found {len(df_unresolved)} unresolved videos. See metadata/unresolved_videos.csv")
        df_unresolved.to_csv(metadata_dir / "unresolved_videos.csv", index=False)
        
    if len(df_videos) == 0:
        logger.error("No valid videos found.")
        sys.exit(1)
        
    logger.info(f"Found {len(df_videos)} properly identified videos.")
    logger.info(df_videos["category"].value_counts())
    
    # Apply max videos limit if set (for small testing)
    if args.max_videos and args.max_videos < len(df_videos):
        # We need to subset carefully to avoid isolating halves of pairs
        # The easiest way is to build identity groups first
        df_videos = build_identity_groups(df_videos)
        unique_groups = list(df_videos["group_id"].unique())
        
        # Pick enough groups to roughly meet max_videos
        selected_groups = []
        video_count = 0
        for g in unique_groups:
            if video_count >= args.max_videos:
                break
            selected_groups.append(g)
            video_count += len(df_videos[df_videos["group_id"] == g])
            
        df_videos = df_videos[df_videos["group_id"].isin(selected_groups)].copy()
        logger.info(
            f"Subsampled to {len(df_videos)} videos ({len(selected_groups)} identity groups) for small test. "
            f"(Note: --max-videos is a soft limit to ensure complete identity components remain intact)"
        )
    
    # 2. Split at Identity Level
    logger.info("Splitting videos into train/val/test using Union-Find components...")
    test_ratio = 1.0 - (args.train_ratio + args.val_ratio)
    train_v, val_v, test_v = source_aware_split(
        df_videos, args.train_ratio, args.val_ratio, test_ratio, seed=42
    )
    
    # 3. Extract Faces per split
    extractor = FaceExtractor(output_size=args.image_size)
    
    logger.info(f"Extracting up to {args.frames_per_video} frames per video...")
    train_meta = extract_faces_for_split(train_v, "train", output_dir, extractor, args.frames_per_video)
    val_meta = extract_faces_for_split(val_v, "val", output_dir, extractor, args.frames_per_video)
    test_meta = extract_faces_for_split(test_v, "test", output_dir, extractor, args.frames_per_video)
    
    # Check Leakage (Final check on extracted frames metadata)
    logger.info("Verifying final extracted frames for leakage...")
    if len(train_meta) > 0 and len(val_meta) > 0 and len(test_meta) > 0:
        check_leakage(train_meta, val_meta, test_meta)
    
    # 4. Save Metadata
    splits_dir = output_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    if len(train_meta) > 0: train_meta.to_csv(splits_dir / "train.csv", index=False)
    if len(val_meta) > 0: val_meta.to_csv(splits_dir / "val.csv", index=False)
    if len(test_meta) > 0: test_meta.to_csv(splits_dir / "test.csv", index=False)
    
    all_meta = pd.concat([train_meta, val_meta, test_meta], ignore_index=True)
    if len(all_meta) > 0:
        all_meta.to_csv(metadata_dir / "metadata.csv", index=False)
        
    logger.info("=" * 60)
    logger.info(f"Preparation complete! Total extracted faces: {len(all_meta)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
