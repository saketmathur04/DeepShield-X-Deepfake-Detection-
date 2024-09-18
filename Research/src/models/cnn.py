"""
DeepShield-X — Custom CNN Baseline

Simple 3-block CNN for deepfake detection.
Architecture is intentionally simple for viva explainability:
    Conv→BN→ReLU→Pool (×3) → GAP → FC → Sigmoid
"""

import torch
import torch.nn as nn


class CustomCNN(nn.Module):
    """
    Baseline CNN with 3 convolutional blocks.

    Architecture:
        Block 1: Conv2d(3→32)  → BatchNorm → ReLU → MaxPool
        Block 2: Conv2d(32→64) → BatchNorm → ReLU → MaxPool
        Block 3: Conv2d(64→128)→ BatchNorm → ReLU → MaxPool
        → Global Average Pooling
        → Dropout
        → FC(128→1)

    This model is intentionally simple:
    - No skip connections (unlike ResNet)
    - No depthwise separable convolutions (unlike MobileNet)
    - Serves as the baseline for comparison experiments
    """

    def __init__(
        self,
        num_classes: int = 1,
        dropout: float = 0.5,
        channels: list = None,
    ):
        super().__init__()

        if channels is None:
            channels = [32, 64, 128]

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2
            nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3
            nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channels[2], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def get_gradcam_target_layer(self):
        """Return the last convolutional layer for Grad-CAM."""
        return self.features[8]  # Last Conv2d (Block 3)


def create_custom_cnn(
    num_classes: int = 1,
    dropout: float = 0.5,
    channels: list = None,
    **kwargs,
) -> CustomCNN:
    """Factory function for CustomCNN."""
    return CustomCNN(
        num_classes=num_classes,
        dropout=dropout,
        channels=channels,
    )
