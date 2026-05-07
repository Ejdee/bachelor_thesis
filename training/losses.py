"""=============================================================================
Project:     Accurate and Robust Localization of Landmarks on a Vehicle
Author:      Adam Běhoun <xbehoua00@vutbr.cz>
Year:        2026
Description: Loss functions for car keypoint training.
============================================================================="""

import torch
import torch.nn as nn


class CarFeaturePunisherLoss(nn.Module):
    """Custom loss function for ALIKED detection head fine-tuning.

    Combines:
        1. L2 loss on verified keypoint regions (weighted by pos_weight)
        2. L1 loss on background regions to suppress false positives (weighted by bg_weight)
        3. Soft L2 penalty on unverified vehicle surface to regularize (weighted by empty_car_weight) 
    """
    def __init__(self, pos_weight=1.0, bg_weight=5.0, empty_car_weight=0.5):
        super().__init__()
        self.pos_weight = pos_weight
        self.bg_weight = bg_weight
        self.empty_car_weight = empty_car_weight

    def forward(self, pred, gt_hmap, train_mask, car_seg):
        """
        Args:
            pred: Predicted score map
            gt_hmap: Ground truth Gaussian heatmap
            train_mask: Valid training region mask
            car_seg: Car segmentation mask
        """
        # Verified keypoint region - L2 toward the Gaussian heatmap target
        pos_mask = (train_mask > 0.5).float() * (gt_hmap > 0.01).float()
        pos_loss = ((pred - gt_hmap) ** 2 * pos_mask).sum() / (pos_mask.sum() + 1e-6)

        # Background - L1 to drive responses to zero
        bg_mask = (car_seg < 0.5).float()
        bg_loss = (torch.abs(pred) * bg_mask).sum() / (bg_mask.sum() + 1e-6)

        # Unverified vehicle surface - soft L2 penalty (regularizer)
        empty_mask = (car_seg > 0.5).float() * (gt_hmap <= 0.01).float()
        empty_loss = (pred**2 * empty_mask).sum() / (empty_mask.sum() + 1e-6)

        return (
            self.pos_weight * pos_loss
            + self.bg_weight * bg_loss
            + self.empty_car_weight * empty_loss
        )
