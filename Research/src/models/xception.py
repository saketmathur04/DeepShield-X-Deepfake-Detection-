"""
DeepShield-X — Xception Model

Xception via timm — particularly relevant for face manipulation detection
(used in the original FaceForensics++ paper).
"""

import torch
import torch.nn as nn
import timm


class XceptionDetector(nn.Module):
    """
    Xception for binary deepfake detection.

    Xception uses depthwise separable convolutions and was shown to be
    effective for facial manipulation detection in:
        Rössler et al., "FaceForensics++: Learning to Detect Manipulated Facial Images"
    """

    def __init__(
        self,
        variant: str = "xception",
        num_classes: int = 1,
        pretrained: bool = True,
        dropout: float = 0.5,
        finetune_strategy: str = "full",
        unfreeze_layers: int = 2,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            variant,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout,
        )

        if finetune_strategy == "frozen":
            for name, param in self.backbone.named_parameters():
                if "fc" not in name and "classifier" not in name:
                    param.requires_grad = False

        elif finetune_strategy == "partial":
            for param in self.backbone.parameters():
                param.requires_grad = False
            if hasattr(self.backbone, "fc"):
                for param in self.backbone.fc.parameters():
                    param.requires_grad = True
            # Unfreeze last blocks
            if hasattr(self.backbone, "block12"):
                for param in self.backbone.block12.parameters():
                    param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_gradcam_target_layer(self):
        """Return the last convolutional layer for Grad-CAM."""
        if hasattr(self.backbone, "act4"):
            return self.backbone.act4
        if hasattr(self.backbone, "block12"):
            return self.backbone.block12
        # Fallback
        return self.backbone.bn4


def create_xception(
    variant: str = "xception",
    num_classes: int = 1,
    pretrained: bool = True,
    dropout: float = 0.5,
    finetune_strategy: str = "full",
    unfreeze_layers: int = 2,
    **kwargs,
) -> XceptionDetector:
    """Factory function for Xception."""
    return XceptionDetector(
        variant=variant,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        finetune_strategy=finetune_strategy,
        unfreeze_layers=unfreeze_layers,
    )
