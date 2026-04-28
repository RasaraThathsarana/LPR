"""Segmentation loss helpers used by the training loop."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossOutput:
    """Container for a composite loss and its individual terms."""

    total: torch.Tensor
    ce: torch.Tensor
    dice: torch.Tensor
    boundary: torch.Tensor


class DiceLoss(nn.Module):
    """Multi-class Dice loss with ignore-index support."""

    def __init__(self, ignore_index: int = 255, smooth: float = 1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)

        valid_mask = target != self.ignore_index
        safe_target = target.clone()
        safe_target[~valid_mask] = 0

        target_one_hot = F.one_hot(safe_target.long(), num_classes=num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()

        valid_mask = valid_mask.unsqueeze(1).float()
        probs = probs * valid_mask
        target_one_hot = target_one_hot * valid_mask

        dims = (0, 2, 3)
        intersection = torch.sum(probs * target_one_hot, dim=dims)
        denominator = torch.sum(probs + target_one_hot, dim=dims)

        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        valid_classes = denominator > 0

        if valid_classes.any():
            return 1.0 - dice[valid_classes].mean()
        return logits.new_tensor(0.0)


class BoundaryLoss(nn.Module):
    """Boundary-aware loss based on label edges and soft prediction gradients."""

    def __init__(self, ignore_index: int = 255, boundary_thickness: int = 1):
        super().__init__()
        self.ignore_index = ignore_index
        self.boundary_thickness = max(1, int(boundary_thickness))

    def _target_boundary(self, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        valid = target != self.ignore_index
        safe_target = target.clone()
        safe_target[~valid] = 0

        # Boundary pixels are positions whose label differs from at least one
        # 4-neighbouring pixel with a valid label.
        boundary = torch.zeros_like(valid, dtype=torch.bool)

        # Horizontal edges
        left_valid = valid[:, :, :-1] & valid[:, :, 1:]
        left_change = safe_target[:, :, :-1] != safe_target[:, :, 1:]
        boundary[:, :, :-1] |= left_change & left_valid
        boundary[:, :, 1:] |= left_change & left_valid

        # Vertical edges
        up_valid = valid[:, :-1, :] & valid[:, 1:, :]
        up_change = safe_target[:, :-1, :] != safe_target[:, 1:, :]
        boundary[:, :-1, :] |= up_change & up_valid
        boundary[:, 1:, :] |= up_change & up_valid

        if self.boundary_thickness > 1:
            kernel_size = 2 * self.boundary_thickness - 1
            boundary = F.max_pool2d(
                boundary.float().unsqueeze(1),
                kernel_size=kernel_size,
                stride=1,
                padding=self.boundary_thickness - 1,
            ).squeeze(1) > 0

        # Only supervise where the ground truth is defined.
        return boundary.float(), valid

    def _prediction_boundary(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits.float(), dim=1)

        diff_h = torch.abs(probs[:, :, 1:, :] - probs[:, :, :-1, :]).sum(dim=1, keepdim=True)
        diff_w = torch.abs(probs[:, :, :, 1:] - probs[:, :, :, :-1]).sum(dim=1, keepdim=True)

        edge_map = torch.zeros_like(probs[:, :1, :, :])
        edge_map[:, :, 1:, :] += diff_h
        edge_map[:, :, :-1, :] += diff_h
        edge_map[:, :, :, 1:] += diff_w
        edge_map[:, :, :, :-1] += diff_w

        # Use raw edge magnitude as logits for an autocast-safe BCE loss.
        return edge_map

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_boundary = self._prediction_boundary(logits)
        target_boundary, boundary_valid = self._target_boundary(target)

        loss = F.binary_cross_entropy_with_logits(
            pred_boundary,
            target_boundary.unsqueeze(1),
            reduction='none',
        )

        valid = boundary_valid.unsqueeze(1).float()
        denom = valid.sum()
        if denom > 0:
            return (loss * valid).sum() / denom
        return logits.new_tensor(0.0)


class CompositeSegmentationLoss(nn.Module):
    """CE + Dice + Boundary composite loss."""

    def __init__(
        self,
        ignore_index: int = 255,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        boundary_weight: float = 1.0,
        dice_smooth: float = 1.0,
        boundary_thickness: int = 1,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index, smooth=dice_smooth)
        self.boundary = BoundaryLoss(ignore_index=ignore_index, boundary_thickness=boundary_thickness)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> LossOutput:
        ce_loss = self.ce(logits, target)
        dice_loss = self.dice(logits, target)
        boundary_loss = self.boundary(logits, target)

        total = (
            self.ce_weight * ce_loss
            + self.dice_weight * dice_loss
            + self.boundary_weight * boundary_loss
        )
        return LossOutput(
            total=total,
            ce=ce_loss,
            dice=dice_loss,
            boundary=boundary_loss,
        )
