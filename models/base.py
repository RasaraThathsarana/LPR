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

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        adapter: Optional[Adapter] = None,
        aux_head: Optional[nn.Module] = None,
        input_norm_cfg: Optional[Dict] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.adapter = adapter
        self.decoder = decoder
        self.aux_head = aux_head
        self.input_norm_cfg = input_norm_cfg

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        features = self.encoder(x)
        if self.adapter is not None:
            features = self.adapter(features)
        
        out = self.decoder(features)
        if return_aux and self.aux_head is not None:
            aux_out = self.aux_head(features)
            # Ensure auxiliary output matches input spatial dimensions
            if aux_out.shape[2:] != x.shape[2:]:
                aux_out = torch.nn.functional.interpolate(aux_out, size=x.shape[2:], mode='bilinear', align_corners=False)
            return out, aux_out
        elif return_aux:
            return out, None
        return out
