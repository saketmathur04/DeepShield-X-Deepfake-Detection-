"""
DeepShield-X — Loss Functions

Binary Cross-Entropy and Focal Loss for deepfake detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def binary_pos_weight(num_negative: int, num_positive: int) -> float:
    """Return BCEWithLogitsLoss's positive-class weight from training counts.

    For this project, REAL is label 0 (negative) and FAKE is label 1
    (positive).  ``pos_weight`` therefore balances the two classes as
    ``num_real / num_fake``.  A value below one is correct when FAKE is the
    majority class: it down-weights FAKE relative to REAL.
    """
    if num_negative < 0 or num_positive <= 0:
        raise ValueError("Class counts must be non-negative and num_positive must be > 0")
    return float(num_negative / num_positive)


class BCEWithLogitsSmoothed(nn.Module):
    """
    Binary Cross-Entropy with optional label smoothing.
    
    Label smoothing helps prevent overconfident predictions and can
    improve calibration.
    """

    def __init__(self, label_smoothing: float = 0.0, pos_weight: torch.Tensor = None):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        targets = targets.view_as(logits)
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight
        )


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    Focuses training on hard-to-classify examples by down-weighting
    well-classified ones.

    Reference: Lin et al., "Focal Loss for Dense Object Detection" (2017)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.view_as(logits)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        return (focal_weight * bce_loss).mean()


def create_loss(
    loss_name: str = "bce",
    label_smoothing: float = 0.0,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    pos_weight: torch.Tensor = None,
) -> nn.Module:
    """
    Factory function for loss creation.

    Args:
        loss_name: "bce" or "focal".
        label_smoothing: Smoothing factor for BCE.
        focal_alpha: Alpha for Focal Loss.
        focal_gamma: Gamma for Focal Loss.
        pos_weight: Class weight for positive class.

    Returns:
        Loss module.
    """
    if loss_name == "bce":
        return BCEWithLogitsSmoothed(
            label_smoothing=label_smoothing,
            pos_weight=pos_weight,
        )
    elif loss_name == "focal":
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    else:
        raise ValueError(f"Unknown loss: {loss_name}. Available: bce, focal")
