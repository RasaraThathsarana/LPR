"""Auxiliary decoder heads for UPerNet-style models."""

import torch
import torch.nn as nn
from typing import List


class AuxiliaryFCNHead(nn.Module):
    """Auxiliary FCN head used by MMSeg UPerNet recipes."""

    def __init__(
        self,
        in_channels: int,
        channels: int = 256,
        num_convs: int = 1,
        concat_input: bool = False,
        num_classes: int = 150,
        dropout_ratio: float = 0.1,
        in_index: int = 2,
        align_corners: bool = False,
    ):
        super().__init__()
        self.in_index = in_index
        self.concat_input = concat_input
        self.align_corners = align_corners
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()

        convs = []
        in_ch = in_channels
        for _ in range(num_convs):
            convs.append(nn.Conv2d(in_ch, channels, kernel_size=3, padding=1, bias=False))
            convs.append(nn.BatchNorm2d(channels))
            convs.append(nn.ReLU(inplace=True))
            in_ch = channels
        self.convs = nn.Sequential(*convs) if convs else nn.Identity()

        if concat_input:
            self.conv_cat = nn.Sequential(
                nn.Conv2d(in_channels + channels, channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv_cat = None

        self.cls_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, features: List[torch.Tensor], output_size=None) -> torch.Tensor:
        x = features[self.in_index]
        shortcut = x
        x = self.convs(x)

        if self.conv_cat is not None:
            x = self.conv_cat(torch.cat([shortcut, x], dim=1))

        x = self.dropout(x)
        x = self.cls_seg(x)

        if output_size is not None:
            x = nn.functional.interpolate(
                x,
                size=output_size,
                mode='bilinear',
                align_corners=self.align_corners,
            )
        return x
