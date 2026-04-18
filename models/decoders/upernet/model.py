"""UPerNet-style decoder implementation."""

import torch
import torch.nn as nn
from typing import List, Sequence
from ...base import Decoder


class UPerNetDecoder(Decoder):
    """UPerNet-style decoder for multi-stage feature fusion."""

    def __init__(
        self,
        in_channels: Sequence[int],
        channels: int = 512,
        num_classes: int = 150,
        dropout_ratio: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio

        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, channels, kernel_size=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
            for in_ch in in_channels
        ])

        self.fuse = nn.Sequential(
            nn.Conv2d(len(in_channels) * channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.cls_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        target_size = features[0].shape[-2:]
        processed = []
        for feature, conv in zip(features, self.fpn_convs):
            processed.append(
                nn.functional.interpolate(
                    conv(feature),
                    size=target_size,
                    mode='bilinear',
                    align_corners=False,
                )
            )

        x = torch.cat(processed, dim=1)
        x = self.fuse(x)
        x = self.dropout(x)
        x = self.cls_seg(x)
        return nn.functional.interpolate(
            x,
            scale_factor=4,
            mode='bilinear',
            align_corners=False,
        )
