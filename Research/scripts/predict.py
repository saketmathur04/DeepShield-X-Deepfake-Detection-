"""
DeepShield-X — Single Image Prediction Script

Usage:
    python scripts/predict.py --image test.jpg --checkpoint path/to/model.pth
    python scripts/predict.py --image test.jpg --model resnet50 --smoke-test
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.device import get_device
from src.utils.logging import setup_logging
from src.inference.predictor import Predictor


def parse_args():
    parser = argparse.ArgumentParser(description="DeepShield-X Prediction")
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--gradcam", action="store_true", help="Generate Grad-CAM")
    parser.add_argument("--output-dir", type=str, default="./experiments/plots/predictions")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging("deepshield")
    device = get_device(args.device)

    image_path = Path(args.image)
    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        sys.exit(1)

    # Create predictor
    predictor = Predictor(
        checkpoint_path=args.checkpoint,
        model_name=args.model,
        device=device,
    )

    if args.gradcam:
        result = predictor.predict_with_gradcam(str(image_path))
        logger.info(f"\nPrediction: {result['prediction']}")
        logger.info(f"Confidence: {result['confidence']:.4f}")
        logger.info(f"Probability: {result['probability']:.4f}")

        # Save Grad-CAM
        from src.explainability.gradcam import GradCAMExplainer
        import numpy as np
        from PIL import Image

        orig_img = np.array(Image.open(image_path).convert("RGB"))
        explainer = GradCAMExplainer(predictor.model, device=device)
        explainer.save_explanation(
            result,
            save_dir=args.output_dir,
            filename=image_path.stem,
            original_image=orig_img,
        )
    else:
        result = predictor.predict(str(image_path))
        logger.info(f"\nPrediction: {result['label']}")
        logger.info(f"Confidence: {result['confidence']:.4f}")
        logger.info(f"Probability: {result['probability']:.4f}")

    # Print as JSON
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
