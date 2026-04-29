"""SegFormer head implementation inspired by MMSegmentation."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...base import Decoder


class SegFormerHead(Decoder):
    def __init__(
        self,
        in_channels: Sequence[int],
        channels: int = 256,
        num_classes: int = 150,
        dropout_ratio: float = 0.1,
        align_corners: bool = False,
    ):
        super().__init__()
        self.in_channels = list(in_channels)
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.align_corners = align_corners

        self.linear_c = nn.ModuleList()
        for in_ch in self.in_channels:
            self.linear_c.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, self.channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(self.channels),
                    nn.ReLU(inplace=True),
                )
            )

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(self.channels * len(self.in_channels), self.channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(self.dropout_ratio) if self.dropout_ratio > 0 else nn.Identity()
        self.cls_seg = nn.Conv2d(self.channels, self.num_classes, kernel_size=1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        target_size = features[0].shape[2:]
        outs = []
        for feature, proj in zip(features, self.linear_c):
            x = proj(feature)
            if x.shape[2:] != target_size:
                x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=self.align_corners)
            outs.append(x)
        x = torch.cat(outs, dim=1)
        x = self.linear_fuse(x)
        x = self.dropout(x)
        x = self.cls_seg(x)
        return x
