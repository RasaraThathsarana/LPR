"""Base classes for encoder-decoder segmentation models."""

import torch
import torch.nn as nn
import inspect
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
        
        # Check if the decoder requires the original image tensor
        sig = inspect.signature(self.decoder.forward)
        self.pass_img_to_decoder = 'img' in sig.parameters

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input tensor with configured mean/std and optional RGB conversion."""
        if not self.input_norm_cfg:
            return x

        mean = self.input_norm_cfg.get('mean')
        std = self.input_norm_cfg.get('std')
        if mean is None or std is None:
            return x

        mean_tensor = torch.tensor(mean, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
        std_tensor = torch.tensor(std, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)

        if self.input_norm_cfg.get('to_rgb', False) and x.shape[1] == 3:
            x = x[:, [2, 1, 0], :, :]

        return (x - mean_tensor) / std_tensor

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        x = self._normalize_input(x)
        features = self.encoder(x)
        encoder_features = features
        if self.adapter is not None:
            features = self.adapter(features)
        
        if self.pass_img_to_decoder:
            out = self.decoder(features, img=x)
        else:
            out = self.decoder(features)
        if return_aux and self.aux_head is not None:
            aux_out = self.aux_head(encoder_features)
            # Ensure auxiliary output matches input spatial dimensions
            if aux_out.shape[2:] != x.shape[2:]:
                aux_out = torch.nn.functional.interpolate(aux_out, size=x.shape[2:], mode='bilinear', align_corners=False)
            return out, aux_out
        elif return_aux:
            return out, None
        return out
