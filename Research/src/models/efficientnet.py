"""
DeepShield-X — EfficientNet Model

EfficientNet-B0/B1/B2 via timm for deepfake detection.
"""

import torch
import torch.nn as nn
import timm


class EfficientNetDetector(nn.Module):
    """
    EfficientNet for binary deepfake detection.

    Uses timm for flexible variant selection (B0, B1, B2).
    EfficientNet is known for strong accuracy-to-parameter efficiency.
    """

    def __init__(
        self,
        variant: str = "efficientnet_b0",
        num_classes: int = 1,
        pretrained: bool = True,
        dropout: float = 0.3,
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

        # Apply fine-tuning strategy
        if finetune_strategy == "frozen":
            for name, param in self.backbone.named_parameters():
                if "classifier" not in name and "fc" not in name:
                    param.requires_grad = False

        elif finetune_strategy == "partial":
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Unfreeze classifier
            if hasattr(self.backbone, "classifier"):
                for param in self.backbone.classifier.parameters():
                    param.requires_grad = True
            # Unfreeze last blocks
            if hasattr(self.backbone, "blocks"):
                blocks = list(self.backbone.blocks)
                for block in blocks[-unfreeze_layers:]:
                    for param in block.parameters():
                        param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_gradcam_target_layer(self):
        """Return the last convolutional feature layer."""
        if hasattr(self.backbone, "conv_head"):
            return self.backbone.conv_head
        # Fallback for different timm versions
        return list(self.backbone.blocks[-1].children())[-1]


def create_efficientnet(
    variant: str = "efficientnet_b0",
    num_classes: int = 1,
    pretrained: bool = True,
    dropout: float = 0.3,
    finetune_strategy: str = "full",
    unfreeze_layers: int = 2,
    **kwargs,
) -> EfficientNetDetector:
    """Factory function for EfficientNet."""
    return EfficientNetDetector(
        variant=variant,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        finetune_strategy=finetune_strategy,
        unfreeze_layers=unfreeze_layers,
    )
