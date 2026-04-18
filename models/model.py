"""Model assembly for encoder-decoder semantic segmentation."""

import importlib
import torch
from typing import Optional

from .base import EncoderDecoderModel, SegmentationModel
from .adapters import build_adapter


def _import_encoder(name: str):
    try:
        return importlib.import_module(f'models.encoders.{name}')
    except ImportError as error:
        raise ValueError(f'Unknown encoder: {name}') from error


def _import_decoder(name: str):
    try:
        return importlib.import_module(f'models.decoders.{name}')
    except ImportError as error:
        raise ValueError(f'Unknown decoder: {name}') from error


def build_model(
    encoder_name: str = 'swin_tiny',
    decoder_name: str = 'upernet',
    adapter_name: Optional[str] = None,
    num_classes: int = 150,
    encoder_kwargs: Optional[dict] = None,
    adapter_kwargs: Optional[dict] = None,
    decoder_kwargs: Optional[dict] = None,
    pretrained: bool = False,
    pretrain_path: Optional[str] = None,
) -> SegmentationModel:
    """Build an encoder-decoder segmentation model."""
    encoder_name = encoder_name.lower()
    decoder_name = decoder_name.lower()
    if adapter_name is not None:
        adapter_name = adapter_name.lower()

    encoder_module = _import_encoder(encoder_name)
    decoder_module = _import_decoder(decoder_name)

    encoder_kwargs = encoder_kwargs or {}
    adapter_kwargs = adapter_kwargs or {}
    decoder_kwargs = decoder_kwargs or {}

    encoder = encoder_module.build_encoder(**encoder_kwargs)
    adapter = None
    if adapter_name:
        adapter = build_adapter(adapter_name, **adapter_kwargs)
    decoder = decoder_module.build_decoder(**decoder_kwargs, num_classes=num_classes)

    model = EncoderDecoderModel(encoder=encoder, decoder=decoder, adapter=adapter)

    if pretrained and pretrain_path:
        checkpoint = torch.load(pretrain_path, map_location='cpu')
        state_dict = checkpoint.get('model', checkpoint)
        model.load_state_dict(state_dict, strict=False)

    return model
