"""
DeepShield-X — MobileNetV3 Model

Lightweight MobileNetV3 for efficiency/deployment experiments.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class MobileNetDetector(nn.Module):
    """
    MobileNetV3 for binary deepfake detection.

    Small variant is ideal for studying deployment-friendly architectures.
    Large variant provides better accuracy at slightly higher cost.
    """

    def __init__(
        self,
        variant: str = "small",
        num_classes: int = 1,
        pretrained: bool = True,
        dropout: float = 0.3,
        finetune_strategy: str = "full",
        unfreeze_layers: int = 2,
    ):
        super().__init__()
        self.variant = variant

        if variant == "small":
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.mobilenet_v3_small(weights=weights)
        else:
            weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
            self.backbone = models.mobilenet_v3_large(weights=weights)

        # Replace classifier
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)

        # Apply fine-tuning strategy
        if finetune_strategy == "frozen":
            for name, param in self.backbone.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False

        elif finetune_strategy == "partial":
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True
            features = list(self.backbone.features)
            for layer in features[-unfreeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_gradcam_target_layer(self):
        """Return the last feature layer for Grad-CAM."""
        return self.backbone.features[-1]


def create_mobilenet(
    variant: str = "small",
    num_classes: int = 1,
    pretrained: bool = True,
    dropout: float = 0.3,
    finetune_strategy: str = "full",
    unfreeze_layers: int = 2,
    **kwargs,
) -> MobileNetDetector:
    """Factory function for MobileNetV3."""
    return MobileNetDetector(
        variant=variant,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        finetune_strategy=finetune_strategy,
        unfreeze_layers=unfreeze_layers,
    )
