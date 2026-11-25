"""
DeepShield-X — Face Extraction Script

Usage:
    python scripts/extract_faces.py --input ./data/raw/faceforensics --output ./data/faces
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logging import setup_logging
from src.data.face_extraction import batch_extract_faces


def parse_args():
    parser = argparse.ArgumentParser(description="DeepShield-X Face Extraction")
    parser.add_argument("--input", type=str, required=True, help="Input directory")
    parser.add_argument("--output", type=str, default="./data/faces", help="Output directory")
    parser.add_argument("--label", type=str, default="real", choices=["real", "fake"])
    parser.add_argument("--frame-interval", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging("deepshield")

    logger.info(f"Extracting faces from: {args.input}")
    total = batch_extract_faces(
        input_dir=args.input,
        output_dir=args.output,
        label=args.label,
        frame_interval=args.frame_interval,
        max_frames_per_video=args.max_frames,
        image_size=args.image_size,
    )
    logger.info(f"Extracted {total} faces → {args.output}")


if __name__ == "__main__":
    main()
