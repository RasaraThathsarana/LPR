"""Simple encoder for segmentation models."""

import torch
import torch.nn as nn
from typing import List, Sequence
from ...base import Encoder


class SimpleEncoder(Encoder):
    """Simple convolutional encoder with multiple output stages."""

    def __init__(
        self,
        in_channels: int = 3,
        channels: Sequence[int] = (64, 128, 256, 512),
    ):
        super().__init__()
        self.stages = nn.ModuleList()
        current = in_channels
        for out_channels in channels:
            block = nn.Sequential(
                nn.Conv2d(current, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            self.stages.append(block)
            current = out_channels

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features
