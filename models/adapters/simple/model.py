"""Simple adapter implementation."""

import torch
import torch.nn as nn
from typing import List, Sequence
from ..base import Adapter


class SimpleAdapter(Adapter):
    """Simple adapter that projects encoder features to target channels."""

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: Sequence[int],
    ):
        super().__init__()
        if len(in_channels) != len(out_channels):
            raise ValueError('in_channels and out_channels must have the same length')

        self.projections = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, kernel_size=1)
            for in_ch, out_ch in zip(in_channels, out_channels)
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        return [proj(feat) for proj, feat in zip(self.projections, features)]
