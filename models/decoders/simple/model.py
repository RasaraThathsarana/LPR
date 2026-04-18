"""Simple segmentation decoder."""

import torch
import torch.nn as nn
from typing import List
from ...base import Decoder


class SimpleDecoder(Decoder):
    """Simple decoder that upsamples the last encoder feature map."""

    def __init__(
        self,
        in_channels: int = 512,
        num_classes: int = 150,
        upsample_scale: int = 4,
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, num_classes, kernel_size=1),
        )
        self.upsample_scale = upsample_scale

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        x = features[-1]
        x = self.head(x)
        return nn.functional.interpolate(
            x,
            scale_factor=self.upsample_scale,
            mode='bilinear',
            align_corners=False,
        )
