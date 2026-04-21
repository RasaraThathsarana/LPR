"""Model assembly for encoder-decoder semantic segmentation."""

import importlib
import torch
from typing import Optional

from .base import EncoderDecoderModel, SegmentationModel
from .adapters import build_adapter
from .aux_decoders import build_auxiliary_head

SWIN_URLS = {
    'swin_tiny': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth',
    'swin_small': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth',
    'swin_base': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth',
    'swin_large': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth'
}


def _count_parameters(module: Optional[torch.nn.Module]):
    """Count total and trainable parameters of a module."""
    if module is None:
        return 0, 0
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def _format_params(num: int) -> str:
    """Format parameter count to millions."""
    return f"{num / 1e6:.2f}M"

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


def translate_checkpoint_state_dict(state_dict: dict) -> dict:
    """Translate MMSegmentation checkpoint keys to the local module layout."""
    new_state_dict = {}
    
    # Check if this is a pure backbone checkpoint from Microsoft
    is_pure_backbone = any(k.startswith('patch_embed.') or k.startswith('layers.') for k in state_dict.keys())
    
    for k, v in state_dict.items():
        if k.startswith('auxiliary_head') or k.startswith('head.'):
            continue
            
        if is_pure_backbone and not k.startswith('encoder.') and not k.startswith('decoder.'):
            k = 'encoder.' + k
            
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
                k = k.replace('.1.conv.', '.1.').replace('.1.bn.', '.2.')
            elif any(x in k for x in ['bottleneck', 'lateral_convs', 'fpn_convs', 'fpn_bottleneck']):
                k = k.replace('.conv.', '.0.').replace('.bn.', '.1.')
        new_state_dict[k] = v

    return new_state_dict


def load_checkpoint_smart(model: torch.nn.Module, state_dict: dict):
    """Load checkpoint with MMSegmentation key translation."""
    new_state_dict = translate_checkpoint_state_dict(state_dict)
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
    use_auxiliary_decoder: bool = True,
    auxiliary_kwargs: Optional[dict] = None,
    input_norm_cfg: Optional[dict] = None,
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
    auxiliary_kwargs = auxiliary_kwargs or {}

    encoder = encoder_module.build_encoder(**encoder_kwargs)
    adapter = None
    if adapter_name:
        adapter = build_adapter(adapter_name, **adapter_kwargs)
    decoder = decoder_module.build_decoder(**decoder_kwargs, num_classes=num_classes)
    aux_head = None
    aux_type = decoder_name
    if use_auxiliary_decoder:
        try:
            aux_config = {'num_classes': num_classes}
            aux_config.update(auxiliary_kwargs)
            aux_type = aux_config.pop('type', decoder_name)
            aux_head = build_auxiliary_head(aux_type, **aux_config)
        except (ValueError, AttributeError) as e:
            print(f"Notice: Auxiliary decoder disabled. ({e})")

    # Print Model Assembly Summary
    enc_total, enc_train = _count_parameters(encoder)
    dec_total, dec_train = _count_parameters(decoder)
    adp_total, adp_train = _count_parameters(adapter)
    aux_total, aux_train = _count_parameters(aux_head)
    
    total_params = enc_total + dec_total + adp_total + aux_total
    total_train = enc_train + dec_train + adp_train + aux_train
    
    print("\n========================================================================")
    print("Model Assembly")
    print("========================================================================")
    print(f"Encoder           : {encoder_name}")
    print(f"  params          : {_format_params(enc_total)} total | {_format_params(enc_train)} trainable")
    print(f"Decoder           : {decoder_name}")
    print(f"  params          : {_format_params(dec_total)} total | {_format_params(dec_train)} trainable")
    if adapter:
        print(f"Adapter           : {adapter_name}")
        print(f"  params          : {_format_params(adp_total)} total | {_format_params(adp_train)} trainable")
    else:
        print("Adapter           : disabled")
    if aux_head:
        print(f"Aux decoder       : {aux_type}")
        print(f"  params          : {_format_params(aux_total)} total | {_format_params(aux_train)} trainable")
    else:
        print("Aux decoder       : disabled")
    print("------------------------------------------------------------------------")
    print(f"Total params      : {_format_params(total_params)}")
    print(f"Trainable params  : {_format_params(total_train)}")
    print("========================================================================\n")

    model = EncoderDecoderModel(
        encoder=encoder,
        decoder=decoder,
        adapter=adapter,
        aux_head=aux_head,
        input_norm_cfg=input_norm_cfg,
    )

    if pretrained:
        state_dict = None
        if pretrain_path:
            checkpoint = torch.load(pretrain_path, map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
        elif encoder_name in SWIN_URLS:
            print(f"Auto-downloading Microsoft Official ImageNet weights for {encoder_name}...")
            checkpoint = torch.hub.load_state_dict_from_url(SWIN_URLS[encoder_name], map_location='cpu')
            state_dict = checkpoint.get('model', checkpoint)
            
        if state_dict is not None:
            load_checkpoint_smart(model, state_dict)

    return model
