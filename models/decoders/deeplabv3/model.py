"""DeepLabV3 decoder implementation inspired by MMSegmentation."""

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


class DeepLabV3Head(Decoder):
    def __init__(
        self,
        in_channels: Sequence[int] | int,
        channels: int = 512,
        num_classes: int = 150,
        dilations: Sequence[int] = (12, 24, 36),
        dropout_ratio: float = 0.1,
        align_corners: bool = False,
    ):
        super().__init__()
        if isinstance(in_channels, (tuple, list)):
            in_channels = in_channels[-1]
        self.in_channels = in_channels
        self.channels = channels
        self.num_classes = num_classes
        self.align_corners = align_corners

        self.aspp = ASPPModule(self.in_channels, self.channels, dilations)
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.cls_seg = nn.Conv2d(self.channels, self.num_classes, kernel_size=1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        x = features[-1]
        x = self.aspp(x)
        x = self.dropout(x)
        x = self.cls_seg(x)
        return x
