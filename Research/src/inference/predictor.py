"""
    
DeepShield-X — Predictor (Inference Module)

API-ready inference module. Independent of Streamlit.
Can later be wrapped in FastAPI without restructuring.

Usage:
    predictor = Predictor(checkpoint_path="path/to/model.pth")
    result = predictor.predict(image)
    # {"label": "FAKE", "probability": 0.974, "confidence": 0.974}
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.models.factory import create_model
from src.data.preprocessing import get_val_transforms

logger = logging.getLogger(__name__)


class Predictor:
    """
    Production-ready inference class.

    Loads a trained model from checkpoint and provides a simple predict() API.
    Designed to be easily wrapped in FastAPI/Flask later.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_name: str = "resnet50",
        model: Optional[nn.Module] = None,
        image_size: int = 224,
        device: Optional[torch.device] = None,
        threshold: float = 0.5,
    ):
        """
        Args:
            checkpoint_path: Path to saved model checkpoint.
            model_name: Model architecture name (used if no checkpoint config).
            model: Pre-loaded model (alternative to checkpoint).
            image_size: Input image size.
            device: Compute device.
            threshold: Classification threshold.
        """
        from src.utils.device import get_device

        self.device = device or get_device()
        self.image_size = image_size
        self.threshold = threshold
        self.transform = get_val_transforms(image_size)

        if model is not None:
            self.model = model.to(self.device)
        elif checkpoint_path:
            self.model = self._load_from_checkpoint(checkpoint_path, model_name)
        else:
            self.model = create_model(model_name, pretrained=True).to(self.device)

        self.model.eval()
        logger.info(f"Predictor ready: device={self.device}, image_size={image_size}")

    def _load_from_checkpoint(self, path: str, fallback_name: str) -> nn.Module:
        """Load model from checkpoint file."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Try to get model name from checkpoint config
        config = checkpoint.get("config", {})
        model_name = config.get("model", {}).get("name", fallback_name)

        model = create_model(model_name, pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)

        logger.info(f"Loaded model '{model_name}' from: {path}")
        return model

    def predict(
        self,
        image: Union[str, np.ndarray, Image.Image],
    ) -> Dict[str, Any]:
        """
        Predict REAL or FAKE for a single image.

        Args:
            image: File path, numpy array (RGB), or PIL Image.

        Returns:
            {
                "label": "FAKE",
                "probability": 0.974,
                "confidence": 0.974
            }
        """
        # Load image
        img_array = self._load_image(image)

        # Apply transforms
        transformed = self.transform(image=img_array)
        img_tensor = transformed["image"].unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            output = self.model(img_tensor).squeeze()
            probability = torch.sigmoid(output).item()

        label = "FAKE" if probability >= self.threshold else "REAL"
        confidence = probability if label == "FAKE" else 1 - probability

        return {
            "label": label,
            "probability": round(probability, 4),
            "confidence": round(confidence, 4),
        }

    def predict_with_gradcam(
        self,
        image: Union[str, np.ndarray, Image.Image],
    ) -> Dict[str, Any]:
        """
        Predict with Grad-CAM explanation.

        Returns prediction + heatmap overlay.
        """
        from src.explainability.gradcam import GradCAMExplainer

        img_array = self._load_image(image)
        img_normalized = img_array.astype(np.float32) / 255.0

        transformed = self.transform(image=img_array)
        img_tensor = transformed["image"]

        explainer = GradCAMExplainer(self.model, device=self.device)
        result = explainer.explain(img_tensor, img_normalized, threshold=self.threshold)

        return result

    def _load_image(self, image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """Load and convert image to standard RGB numpy array."""
        if isinstance(image, str) or isinstance(image, Path):
            img = Image.open(str(image)).convert("RGB")
            return np.array(img)
        elif isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif image.shape[2] == 3:
                # Array is 3-channel RGB
                pass
            return image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    @torch.no_grad()
    def predict_batch(
        self,
        images: list,
    ) -> list:
        """Predict on a batch of images."""
        return [self.predict(img) for img in images]
