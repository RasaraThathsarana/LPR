"""PSP head implementation inspired by MMSegmentation."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...base import Decoder


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pool_scales: Sequence[int]):
        super().__init__()
        self.psp_modules = nn.ModuleList()
        for pool_scale in pool_scales:
            self.psp_modules.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + len(pool_scales) * out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [x]
        for psp_module in self.psp_modules:
            out = psp_module(x)
            outs.append(F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False))
        return self.bottleneck(torch.cat(outs, dim=1))


class PSPHead(Decoder):
    def __init__(
        self,
        in_channels: Sequence[int] | int,
        channels: int = 512,
        num_classes: int = 150,
        pool_scales: Sequence[int] = (1, 2, 3, 6),
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

        self.ppm = PyramidPoolingModule(self.in_channels, self.channels, pool_scales)
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.cls_seg = nn.Conv2d(self.channels, self.num_classes, kernel_size=1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        x = features[-1]
        x = self.ppm(x)
        x = self.dropout(x)
        x = self.cls_seg(x)
        return x
