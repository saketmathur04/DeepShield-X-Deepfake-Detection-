"""
DeepShield-X — Vision Transformer (ViT) Model

ViT via timm for comparing transformer-based vs CNN-based deepfake detection.
"""

import torch
import torch.nn as nn
import timm


class ViTDetector(nn.Module):
    """
    Vision Transformer for binary deepfake detection.

    Uses ViT-Small/16 by default (22M params) for reasonable computation.
    Can be switched to ViT-Base/16 (86M params) for higher capacity.

    ViT processes images as sequences of patches, enabling comparison of:
    - CNN local feature extraction vs Transformer global attention patterns.
    """

    def __init__(
        self,
        variant: str = "vit_small_patch16_224",
        num_classes: int = 1,
        pretrained: bool = True,
        dropout: float = 0.1,
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
                if "head" not in name:
                    param.requires_grad = False

        elif finetune_strategy == "partial":
            for param in self.backbone.parameters():
                param.requires_grad = False
            if hasattr(self.backbone, "head"):
                for param in self.backbone.head.parameters():
                    param.requires_grad = True
            # Unfreeze last transformer blocks
            if hasattr(self.backbone, "blocks"):
                blocks = list(self.backbone.blocks)
                for block in blocks[-unfreeze_layers:]:
                    for param in block.parameters():
                        param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_gradcam_target_layer(self):
        """
        Return target layer for Grad-CAM on ViT.
        We use the patch embedding projection layer (Conv2d) which preserves 2D
        spatial dimensions and generates valid Grad-CAM heatmaps without token shape errors.
        """
        if hasattr(self.backbone, "patch_embed") and hasattr(self.backbone.patch_embed, "proj"):
            return self.backbone.patch_embed.proj
        if hasattr(self.backbone, "blocks"):
            return self.backbone.blocks[-1].norm1
        return self.backbone.norm


def create_vit(
    variant: str = "vit_small_patch16_224",
    num_classes: int = 1,
    pretrained: bool = True,
    dropout: float = 0.1,
    finetune_strategy: str = "full",
    unfreeze_layers: int = 2,
    **kwargs,
) -> ViTDetector:
    """Factory function for Vision Transformer."""
    return ViTDetector(
        variant=variant,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        finetune_strategy=finetune_strategy,
        unfreeze_layers=unfreeze_layers,
    )
