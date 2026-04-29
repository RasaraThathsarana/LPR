"""FPN head implementation inspired by MMSegmentation."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...base import Decoder


class FPNHead(Decoder):
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
        self.align_corners = align_corners

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_ch in self.in_channels:
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, self.channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(self.channels),
                    nn.ReLU(inplace=True),
                )
            )
            self.fpn_convs.append(
                nn.Sequential(
                    nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(self.channels),
                    nn.ReLU(inplace=True),
                )
            )

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(len(self.in_channels) * self.channels, self.channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.cls_seg = nn.Conv2d(self.channels, self.num_classes, kernel_size=1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        laterals = [lateral_conv(feat) for lateral_conv, feat in zip(self.lateral_convs, features)]

        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners,
            )

        outs = [fpn_conv(lateral) for fpn_conv, lateral in zip(self.fpn_convs, laterals)]
        for i in range(1, len(outs)):
            outs[i] = F.interpolate(
                outs[i],
                size=outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners,
            )

        x = torch.cat(outs, dim=1)
        x = self.fuse_conv(x)
        x = self.dropout(x)
        x = self.cls_seg(x)
        return x
