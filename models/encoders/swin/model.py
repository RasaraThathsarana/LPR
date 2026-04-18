"""Swin-like encoder implementation."""

import torch
import torch.nn as nn
from typing import List, Optional, Sequence
from ...base import Encoder


class SwinEncoder(Encoder):
    """Swin-style encoder that produces stage-level features."""

    def __init__(
        self,
        in_channels: int = 3,
        embed_dims: int = 96,
        depths: Sequence[int] = (2, 2, 6, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        patch_size: int = 4,
        drop_path_rate: float = 0.2,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        current = in_channels
        for stage_idx, out_channels in enumerate(
            [embed_dims, embed_dims * 2, embed_dims * 4, embed_dims * 8]
        ):
            self.layers.append(self._make_stage(current, out_channels, depths[stage_idx]))
            current = out_channels

    def _make_stage(self, in_channels: int, out_channels: int, depth: int) -> nn.Sequential:
        blocks = []
        for _ in range(depth):
            blocks.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ])
            in_channels = out_channels
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return features
