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
        pool_scales: Sequence[int] = (1, 2, 3, 6),
        align_corners: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.align_corners = align_corners

        # Pyramid Pooling Module (PPM) on the deepest feature map
        self.psp_modules = nn.ModuleList()
        for pool_scale in pool_scales:
            self.psp_modules.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    nn.Conv2d(self.in_channels[-1], self.channels, 1, bias=False),
                    nn.BatchNorm2d(self.channels),
                    nn.ReLU(inplace=True)
                )
            )

        # Bottleneck after PPM
        self.bottleneck = nn.Sequential(
            nn.Conv2d(self.in_channels[-1] + len(pool_scales) * self.channels, self.channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True)
        )

        # FPN Lateral and Output Convs
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for in_ch in self.in_channels[:-1]:  # Skip the deepest feature map
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, self.channels, 1, bias=False),
                    nn.BatchNorm2d(self.channels),
                    nn.ReLU(inplace=True)
                )
            )
            self.fpn_convs.append(
                nn.Sequential(
                    nn.Conv2d(self.channels, self.channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(self.channels),
                    nn.ReLU(inplace=True)
                )
            )

        # Final fusion bottleneck
        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(len(self.in_channels) * self.channels, self.channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        
        self.cls_seg = nn.Conv2d(self.channels, self.num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(self.dropout_ratio) if self.dropout_ratio > 0 else nn.Identity()

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        # 1. Apply PPM on the deepest feature map
        psp_outs = [features[-1]]
        for psp_mod in self.psp_modules:
            psp_out = psp_mod(features[-1])
            psp_outs.append(
                nn.functional.interpolate(
                    psp_out,
                    size=features[-1].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners
                )
            )
        psp_outs = torch.cat(psp_outs, dim=1)
        f = self.bottleneck(psp_outs)

        # 2. Build laterals
        laterals = [
            lateral_conv(features[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        laterals.append(f)

        # 3. Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + nn.functional.interpolate(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners
            )

        # 4. Apply FPN post-fusion convolutions (only to shallower features)
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(len(laterals) - 1)
        ]
        fpn_outs.append(laterals[-1])

        # 5. Resize all FPN features to the highest resolution (features[0])
        target_size = fpn_outs[0].shape[2:]
        for i in range(1, len(fpn_outs)):
            fpn_outs[i] = nn.functional.interpolate(
                fpn_outs[i],
                size=target_size,
                mode='bilinear',
                align_corners=self.align_corners
            )

        # 6. Concatenate, fuse, and output
        x = torch.cat(fpn_outs, dim=1)
        x = self.fpn_bottleneck(x)
        x = self.dropout(x)
        x = self.cls_seg(x)
        
        # Upsample to the original input image size
        return nn.functional.interpolate(
            x,
            scale_factor=4,
            mode='bilinear',
            align_corners=self.align_corners,
        )
