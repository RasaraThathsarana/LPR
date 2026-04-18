"""Base classes for encoder-decoder segmentation models."""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional


class SegmentationModel(nn.Module):
    """Base semantic segmentation model."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Encoder(nn.Module):
    """Base encoder for feature extraction."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        raise NotImplementedError


class Decoder(nn.Module):
    """Base decoder for segmentation heads."""

    def __init__(self):
        super().__init__()

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError


class Adapter(nn.Module):
    """Base adapter for encoder-decoder interface."""

    def __init__(self):
        super().__init__()

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        raise NotImplementedError


class EncoderDecoderModel(SegmentationModel):
    """Encoder-decoder segmentation model with optional adapter."""

    def __init__(self, encoder: Encoder, decoder: Decoder, adapter: Optional[Adapter] = None):
        super().__init__()
        self.encoder = encoder
        self.adapter = adapter
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        if self.adapter is not None:
            features = self.adapter(features)
        return self.decoder(features)
