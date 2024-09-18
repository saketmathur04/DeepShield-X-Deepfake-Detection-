"""
DeepShield-X — Preprocessing & Augmentation

Image preprocessing pipeline with configurable Albumentations augmentation.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

logger = logging.getLogger(__name__)


def get_train_transforms(
    image_size: int = 224,
    augmentation_config: Optional[Dict] = None,
) -> A.Compose:
    """
    Build training transform pipeline.

    Args:
        image_size: Target image size.
        augmentation_config: Dict from config.augmentation.

    Returns:
        Albumentations Compose pipeline.
    """
    transforms_list = [
        A.Resize(image_size, image_size),
    ]

    # Initialize cfg deterministically so normalization always works
    cfg = augmentation_config if augmentation_config else {}

    if cfg.get("enabled", False):

        if cfg.get("horizontal_flip", True):
            transforms_list.append(A.HorizontalFlip(p=0.5))

        rotation = cfg.get("rotation_limit", 15)
        if rotation > 0:
            transforms_list.append(A.Rotate(limit=rotation, p=0.5))

        brightness = cfg.get("brightness_limit", 0.2)
        contrast = cfg.get("contrast_limit", 0.2)
        if brightness > 0 or contrast > 0:
            transforms_list.append(
                A.RandomBrightnessContrast(
                    brightness_limit=brightness,
                    contrast_limit=contrast,
                    p=0.5,
                )
            )

        blur_cfg = cfg.get("gaussian_blur", {})
        if isinstance(blur_cfg, dict) and blur_cfg.get("enabled", False):
            transforms_list.append(
                A.GaussianBlur(
                    blur_limit=blur_cfg.get("blur_limit", 3),
                    p=0.3,
                )
            )

        jpeg_cfg = cfg.get("jpeg_compression", {})
        if isinstance(jpeg_cfg, dict) and jpeg_cfg.get("enabled", False):
            transforms_list.append(
                A.ImageCompression(
                    quality_lower=jpeg_cfg.get("quality_lower", 60),
                    quality_upper=jpeg_cfg.get("quality_upper", 100),
                    p=0.3,
                )
            )

        color_cfg = cfg.get("color_jitter", {})
        if isinstance(color_cfg, dict) and color_cfg.get("enabled", False):
            transforms_list.append(
                A.HueSaturationValue(
                    hue_shift_limit=int(color_cfg.get("hue_limit", 0.05) * 180),
                    sat_shift_limit=int(color_cfg.get("saturation_limit", 0.2) * 255),
                    val_shift_limit=20,
                    p=0.3,
                )
            )

        crop_cfg = cfg.get("random_crop", {})
        if isinstance(crop_cfg, dict) and crop_cfg.get("enabled", False):
            scale = crop_cfg.get("scale", [0.8, 1.0])
            transforms_list.append(
                A.RandomResizedCrop(
                    size=(image_size, image_size),
                    scale=tuple(scale),
                    p=0.5,
                )
            )

    # Normalization (ImageNet defaults) — cfg is always defined
    normalize = cfg.get("normalize", {})
    mean = normalize.get("mean", [0.485, 0.456, 0.406])
    std = normalize.get("std", [0.229, 0.224, 0.225])

    transforms_list.extend([
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    return A.Compose(transforms_list)


def get_val_transforms(
    image_size: int = 224,
    normalize_config: Optional[Dict] = None,
) -> A.Compose:
    """
    
    Build validation/test transform pipeline (no augmentation).

    Args:
        image_size: Target image size.
        normalize_config: Dict with mean and std.

    Returns:
        Albumentations Compose pipeline.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if normalize_config:
        mean = normalize_config.get("mean", mean)
        std = normalize_config.get("std", std)

    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def apply_jpeg_compression(image: np.ndarray, quality: int) -> np.ndarray:
    """Apply JPEG compression at a specific quality level."""
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, encoded = cv2.imencode(".jpg", image, encode_params)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def apply_gaussian_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """Apply Gaussian blur with a specific kernel size."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def apply_gaussian_noise(image: np.ndarray, sigma: float) -> np.ndarray:
    """Add Gaussian noise with a specific standard deviation."""
    noise = np.random.normal(0, sigma * 255, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)
