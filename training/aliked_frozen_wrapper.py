"""=============================================================================
Project:     Accurate and Robust Localization of Landmarks on a Vehicle
Author:      Adam Běhoun <xbehoua00@vutbr.cz>
Year:        2026
Description: Model wrappers for car keypoint training.
============================================================================="""

import sys
from pathlib import Path

# ALIKED source lives in final/ALIKED/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ALIKED"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.aliked import ALIKED


class TrainableALIKED(nn.Module):
    """
    ALIKED model wrapper with frozen backbone and descriptor head.
    
    Only the score head is trainable - backbone and descriptor extraction are frozen.
    """
    def __init__(self, device="cuda"):
        super().__init__()
        self.net = ALIKED(
            model_name="aliked-n32", device=device, top_k=-1, scores_th=0.0, n_limit=0
        )

        # Freeze all parameters
        for param in self.net.parameters():
            param.requires_grad = False

        # Unfreeze only score head
        for name, param in self.net.named_parameters():
            if "score_head" in name:
                param.requires_grad = True

        trainable = [n for n, p in self.net.named_parameters() if p.requires_grad]
        print(f"Trainable parameters ({len(trainable)}):")
        for n in trainable:
            print(f"  {n}")

    def forward(self, x):
        feature_map, score_map = self.net.extract_dense_map(x)
        desc_map = F.normalize(feature_map, p=2, dim=1)
        h, w = x.shape[2] // 8, x.shape[3] // 8
        desc_map = F.interpolate(
            desc_map, size=(h, w), mode="bilinear", align_corners=False
        )
        score_map = F.interpolate(
            score_map,
            size=(x.shape[2], x.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return score_map, desc_map
