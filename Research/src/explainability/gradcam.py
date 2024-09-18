"""
    
DeepShield-X — Grad-CAM Explainability

Generates Grad-CAM heatmap overlays to explain model predictions.
Supports both CNN and ViT architectures.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

logger = logging.getLogger(__name__)

# Conditional import — pytorch-grad-cam may not be installed
try:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    logger.warning("pytorch-grad-cam not installed. Install with: pip install pytorch-grad-cam")


class GradCAMExplainer:
    """
    Grad-CAM explainability wrapper.

    Generates heatmap visualizations showing which image regions
    contributed most to the model's prediction.

    Usage:
        explainer = GradCAMExplainer(model, target_layer)
        result = explainer.explain(image_tensor, original_image)
        # result contains: heatmap, overlay, prediction, confidence
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        method: str = "gradcam",
        device: torch.device = None,
    ):
        """
        Args:
            model: Trained model.
            target_layer: Layer to compute Grad-CAM on. If None, auto-detect.
            method: "gradcam" or "gradcam++".
            device: Compute device.
        """
        if not GRADCAM_AVAILABLE:
            raise ImportError("pytorch-grad-cam is required. Install: pip install pytorch-grad-cam")

        self.model = model
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

        # Get target layer and reshape transform (supports both CNNs and ViT)
        self.reshape_transform = None
        if target_layer is None:
            target_layer, self.reshape_transform = self._auto_detect_layer_and_transform(model)

        target_layers = [target_layer]

        # Create CAM
        CAMClass = GradCAMPlusPlus if method == "gradcam++" else GradCAM
        if self.reshape_transform is not None:
            self.cam = CAMClass(model=model, target_layers=target_layers, reshape_transform=self.reshape_transform)
        else:
            self.cam = CAMClass(model=model, target_layers=target_layers)

        logger.info(f"GradCAM initialized with method={method}, layer={target_layer}")

    def _auto_detect_layer_and_transform(self, model: nn.Module) -> Tuple[nn.Module, Optional[Any]]:
        """Auto-detect the appropriate target layer and reshape transform for CNNs and ViTs."""
        if hasattr(model, "get_gradcam_target_layer"):
            return model.get_gradcam_target_layer(), None

        # Check for Vision Transformer (ViT)
        if hasattr(model, "backbone") and hasattr(model.backbone, "patch_embed"):
            if hasattr(model.backbone.patch_embed, "proj"):
                return model.backbone.patch_embed.proj, None

        # Fallback: find the last Conv2d
        last_conv = None
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module

        if last_conv is not None:
            return last_conv, None

        raise ValueError(
            "Cannot auto-detect Grad-CAM target layer. "
            "Implement get_gradcam_target_layer() in your model."
        )

    def _auto_detect_layer(self, model: nn.Module) -> nn.Module:
        layer, _ = self._auto_detect_layer_and_transform(model)
        return layer

    @torch.no_grad()
    def explain(
        self,
        image_tensor: torch.Tensor,
        original_image: Optional[np.ndarray] = None,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Generate Grad-CAM explanation for a single image.

        Args:
            image_tensor: Preprocessed image tensor [1, C, H, W] or [C, H, W].
            original_image: Original RGB image as numpy array [H, W, 3] in [0, 1].
            threshold: Decision threshold for FAKE vs REAL label.

        Returns:
            Dict with heatmap, overlay, prediction, and confidence.
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(self.device)

        # Get prediction
        self.model.eval()
        with torch.enable_grad():
            output = self.model(image_tensor).squeeze()
        
        probability = torch.sigmoid(output).item()
        label = "FAKE" if probability >= threshold else "REAL"
        confidence = probability if label == "FAKE" else 1.0 - probability

        # Generate Grad-CAM
        targets = [BinaryClassifierOutputTarget(1)]  # Target: FAKE class

        with torch.enable_grad():
            grayscale_cam = self.cam(input_tensor=image_tensor, targets=targets)
        
        grayscale_cam = grayscale_cam[0, :]  # [H, W]

        # Create overlay if original image provided
        overlay = None
        if original_image is not None:
            if original_image.max() > 1.0:
                original_image = original_image / 255.0
            # Resize cam to match original image
            h, w = original_image.shape[:2]
            cam_resized = cv2.resize(grayscale_cam, (w, h))
            overlay = show_cam_on_image(original_image.astype(np.float32), cam_resized, use_rgb=True)

        return {
            "heatmap": grayscale_cam,
            "overlay": overlay,
            "prediction": label,
            "probability": round(probability, 4),
            "confidence": round(confidence, 4),
        }

    def explain_batch(
        self,
        image_tensors: torch.Tensor,
        original_images: Optional[List[np.ndarray]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate Grad-CAM explanations for a batch of images."""
        results = []
        for i in range(image_tensors.size(0)):
            orig = original_images[i] if original_images else None
            result = self.explain(image_tensors[i], orig)
            results.append(result)
        return results

    def save_explanation(
        self,
        result: Dict[str, Any],
        save_dir: str,
        filename: str = "explanation",
        original_image: Optional[np.ndarray] = None,
    ) -> Dict[str, str]:
        """
        Save Grad-CAM visualization to disk.

        Saves:
            - Original image
            - Heatmap
            - Overlay (original + heatmap)

        Returns:
            Dict mapping output type to file path.
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        # Save heatmap
        heatmap = result["heatmap"]
        heatmap_colored = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        heatmap_path = save_path / f"{filename}_heatmap.png"
        cv2.imwrite(str(heatmap_path), heatmap_colored)
        saved_files["heatmap"] = str(heatmap_path)

        # Save overlay
        if result["overlay"] is not None:
            overlay_path = save_path / f"{filename}_overlay.png"
            overlay_bgr = cv2.cvtColor(result["overlay"], cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(overlay_path), overlay_bgr)
            saved_files["overlay"] = str(overlay_path)

        # Save original
        if original_image is not None:
            orig_path = save_path / f"{filename}_original.png"
            if original_image.max() <= 1.0:
                original_image = (original_image * 255).astype(np.uint8)
            orig_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(orig_path), orig_bgr)
            saved_files["original"] = str(orig_path)

        # Save prediction info
        import json
        info_path = save_path / f"{filename}_info.json"
        info = {
            "prediction": result["prediction"],
            "probability": result["probability"],
            "confidence": result["confidence"],
        }
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)
        saved_files["info"] = str(info_path)

        logger.info(
            f"Explanation saved: {result['prediction']} "
            f"(conf={result['confidence']:.4f}) → {save_dir}"
        )

        return saved_files
