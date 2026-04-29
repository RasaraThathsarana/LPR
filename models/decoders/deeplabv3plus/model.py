"""DeepLabV3+ decoder implementation inspired by MMSegmentation."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...base import Decoder


class ASPPModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        dilations: Sequence[int] = (12, 24, 36),
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(
            nn.Sequential(
                nn.Conv2d(in_channels, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
        )
        for dilation in dilations:
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        channels,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                        bias=False,
                    ),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
            )
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d((len(dilations) + 2) * channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [conv(x) for conv in self.convs]
        pooled = self.image_pool(x)
        pooled = F.interpolate(pooled, size=x.shape[2:], mode='bilinear', align_corners=False)
        outs.append(pooled)
        return self.bottleneck(torch.cat(outs, dim=1))


class DeepLabV3PlusHead(Decoder):
    def __init__(
        self,
        in_channels: Sequence[int],
        channels: int = 256,
        num_classes: int = 150,
        dilations: Sequence[int] = (12, 24, 36),
        dropout_ratio: float = 0.1,
        align_corners: bool = False,
        low_level_index: int = 0,
        low_level_channels: int = 48,
    ):
        super().__init__()
        if isinstance(in_channels, (tuple, list)):
            self.in_channels = list(in_channels)
        else:
            self.in_channels = [in_channels]
        self.channels = channels
        self.num_classes = num_classes
        self.align_corners = align_corners
        self.low_level_index = low_level_index
        self.low_level_channels = low_level_channels

        self.aspp = ASPPModule(self.in_channels[-1], self.channels, dilations)
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(self.in_channels[self.low_level_index], self.low_level_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.low_level_channels),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(self.channels + self.low_level_channels, self.channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.cls_seg = nn.Conv2d(self.channels, self.num_classes, kernel_size=1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        x = self.aspp(features[-1])
        low_level_feat = self.low_level_conv(features[self.low_level_index])
        x = F.interpolate(x, size=low_level_feat.shape[2:], mode='bilinear', align_corners=self.align_corners)
        x = torch.cat([x, low_level_feat], dim=1)
        x = self.fuse(x)
        x = self.dropout(x)
        x = self.cls_seg(x)
        return x
