"""FCN head implementation inspired by MMSegmentation."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...base import Decoder


class FCNHead(Decoder):
    def __init__(
        self,
        in_channels: Sequence[int] | int,
        channels: int = 512,
        num_convs: int = 1,
        concat_input: bool = False,
        num_classes: int = 150,
        dropout_ratio: float = 0.1,
        in_index: int = -1,
        align_corners: bool = False,
    ):
        super().__init__()
        if isinstance(in_channels, (tuple, list)):
            self.in_channels = list(in_channels)
        else:
            self.in_channels = [in_channels]

        self.channels = channels
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.num_classes = num_classes
        self.in_index = in_index
        self.align_corners = align_corners

        in_ch = self.in_channels[self.in_index]
        convs = []
        for _ in range(self.num_convs):
            convs.append(nn.Conv2d(in_ch, self.channels, kernel_size=3, padding=1, bias=False))
            convs.append(nn.BatchNorm2d(self.channels))
            convs.append(nn.ReLU(inplace=True))
            in_ch = self.channels

        self.convs = nn.Sequential(*convs) if convs else nn.Identity()

        if self.concat_input:
            self.conv_cat = nn.Sequential(
                nn.Conv2d(self.in_channels[self.in_index] + self.channels, self.channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(self.channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv_cat = None

        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.cls_seg = nn.Conv2d(self.channels, self.num_classes, kernel_size=1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        x = features[self.in_index]
        identity = x
        x = self.convs(x)
        if self.conv_cat is not None:
            x = self.conv_cat(torch.cat([identity, x], dim=1))
        x = self.dropout(x)
        x = self.cls_seg(x)
        return x
