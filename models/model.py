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


def load_checkpoint_smart(model: torch.nn.Module, state_dict: dict):
    """Load checkpoint with MMSegmentation key translation."""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('auxiliary_head'):
            continue
            
        if k.startswith('backbone.'):
            k = k.replace('backbone.', 'encoder.')
            k = k.replace('.stages.', '.layers.')
            k = k.replace('.attn.w_msa.', '.attn.')
            k = k.replace('patch_embed.projection.', 'patch_embed.proj.')
            k = k.replace('.ffn.layers.0.0.', '.mlp.fc1.')
            k = k.replace('.ffn.layers.1.', '.mlp.fc2.')
            
        elif k.startswith('decode_head.'):
            k = k.replace('decode_head.', 'decoder.')
            k = k.replace('conv_seg.', 'cls_seg.')
            if 'psp_modules' in k:
                k = k.replace('.conv.', '.1.').replace('.bn.', '.2.')
            elif any(x in k for x in ['bottleneck', 'lateral_convs', 'fpn_convs', 'fpn_bottleneck']):
                k = k.replace('.conv.', '.0.').replace('.bn.', '.1.')
        new_state_dict[k] = v
        
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing or unexpected:
        print(f"Warning: Loaded with {len(missing)} missing keys and {len(unexpected)} unexpected keys.")

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
        state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
        load_checkpoint_smart(model, state_dict)

    return model
