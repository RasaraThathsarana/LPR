"""OCR head implementation inspired by MMSegmentation."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...base import Decoder


class SpatialGatherModule(nn.Module):
    def __init__(self, num_classes: int, scale: int = 1):
        super().__init__()
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, feats: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        B, C, H, W = probs.shape
        if feats.shape[2:] != probs.shape[2:]:
            feats = F.interpolate(feats, size=probs.shape[2:], mode='bilinear', align_corners=False)

        probs = probs.view(B, C, -1)
        feats = feats.view(B, feats.shape[1], -1)
        probs = F.softmax(probs, dim=2)
        context = torch.bmm(feats, probs.permute(0, 2, 1))
        return context


class ObjectAttentionBlock(nn.Module):
    def __init__(self, in_channels: int, key_channels: int, value_channels: int):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, key_channels, kernel_size=1, bias=False)
        self.key_conv = nn.Conv1d(value_channels, key_channels, kernel_size=1, bias=False)
        self.value_conv = nn.Conv1d(value_channels, value_channels, kernel_size=1, bias=False)
        self.project = nn.Conv2d(value_channels, in_channels, kernel_size=1, bias=False)
        self.scale = key_channels ** -0.5

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)
        key = self.key_conv(context)
        value = self.value_conv(context)
        sim_map = torch.matmul(query, key) * self.scale
        attn = F.softmax(sim_map, dim=-1)
        out = torch.matmul(attn, value.permute(0, 2, 1))
        out = out.permute(0, 2, 1).view(B, -1, H, W)
        out = self.project(out)
        return out


class OCRHead(Decoder):
    def __init__(
        self,
        in_channels: Sequence[int] | int,
        channels: int = 512,
        key_channels: int | None = None,
        value_channels: int | None = None,
        num_classes: int = 150,
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

        self.conv3x3_ocr = nn.Sequential(
            nn.Conv2d(self.in_channels, self.channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_ratio),
        )
        self.conv3x3_cls = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
        )
        self.cls_coarse = nn.Conv2d(self.channels, self.num_classes, kernel_size=1)
        self.cls_seg = nn.Conv2d(self.channels, self.num_classes, kernel_size=1)

        self.spatial_gather = SpatialGatherModule(self.num_classes)
        self.object_attention = ObjectAttentionBlock(
            in_channels=self.channels,
            key_channels=key_channels or self.channels // 2,
            value_channels=value_channels or self.channels,
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(self.channels + self.channels, self.channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        x = features[-1]
        feats = self.conv3x3_ocr(x)
        probs = self.cls_coarse(self.conv3x3_cls(feats))
        context = self.spatial_gather(feats, probs)
        ocr_feats = self.object_attention(feats, context)
        x = self.bottleneck(torch.cat([feats, ocr_feats], dim=1))
        x = self.cls_seg(x)
        return x
