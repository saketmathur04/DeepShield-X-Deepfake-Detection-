"""
DeepShield-X — ResNet50 Model

ImageNet-pretrained ResNet50 with configurable fine-tuning.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50Detector(nn.Module):
    """
    ResNet-50 for binary deepfake detection.

    Uses ImageNet pretrained weights and replaces the classification head
    with a binary output. Supports frozen/partial/full fine-tuning.
    """

    def __init__(
        self,
        num_classes: int = 1,
        pretrained: bool = True,
        dropout: float = 0.5,
        finetune_strategy: str = "full",
        unfreeze_layers: int = 2,
    ):
        super().__init__()

        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=weights)

        # Replace classification head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

        # Apply fine-tuning strategy
        self._apply_finetune_strategy(finetune_strategy, unfreeze_layers)

    def _apply_finetune_strategy(self, strategy: str, unfreeze_layers: int) -> None:
        if strategy == "frozen":
            for name, param in self.backbone.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False

        elif strategy == "partial":
            # Freeze all, then unfreeze last N layers
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone.fc.parameters():
                param.requires_grad = True

            layers = [self.backbone.layer4, self.backbone.layer3,
                      self.backbone.layer2, self.backbone.layer1]
            for layer in layers[:unfreeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = True

        # "full" — all parameters trainable (default)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_gradcam_target_layer(self):
        """Return layer4 for Grad-CAM visualization."""
        return self.backbone.layer4[-1]


def create_resnet50(
    num_classes: int = 1,
    pretrained: bool = True,
    dropout: float = 0.5,
    finetune_strategy: str = "full",
    unfreeze_layers: int = 2,
    **kwargs,
) -> ResNet50Detector:
    """Factory function for ResNet50."""
    return ResNet50Detector(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        finetune_strategy=finetune_strategy,
        unfreeze_layers=unfreeze_layers,
    )
