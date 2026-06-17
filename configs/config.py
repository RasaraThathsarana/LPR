"""
Training configurations for Swin UPerNet models.

These configurations replicate MMSegmentation's settings for:
- Swin Tiny
- Swin Small
- Swin Base
- Swin Large
"""

from __future__ import annotations

import copy


def _deep_merge_dicts(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged

DEFAULT_CONFIG_NAME = 'swin_base'
# Global training hyperparameters applied to all models
GLOBAL_BATCH_SIZE = 4
GLOBAL_ACCUMULATION_STEPS = 4
GLOBAL_LEARNING_RATE = 5e-5

# Base configuration common to all variants
BASE_CONFIG = {
    'num_classes': 150,
    'dataset': 'ade20k',
    'data_root': 'data/ade/ADEChallengeData2016',
    'crop_size': (512, 512),
    'data_preprocessor': {
        'mean': [123.675, 116.28, 103.53],
        'std': [58.395, 57.12, 57.375],
        'to_rgb': False,
    },
    
    # Training settings (matches MMSeg schedule_160k.py)
    'train_cfg': {
        'max_iters': 50000,
        'val_interval': 1000,
    },
    
    # Data loading
    'num_workers': 4,
    'pin_memory': True,
    'val_batch_size': None,
    
    'batch_size': GLOBAL_BATCH_SIZE,
    'accumulation_steps': GLOBAL_ACCUMULATION_STEPS,

    # Segmentation loss
    'loss': {
        'ignore_index': 255,
        'ce_weight': 1.0,
        'dice_weight': 1.0,
        'boundary_weight': 1.0,
        'aux_ce_weight': 1.0,
        'aux_dice_weight': 1.0,
        'aux_boundary_weight': 1.0,
        'dice_smooth': 1.0,
        'boundary_thickness': 1,
        'aux_boundary_thickness': 1,
    },
    
    # Logging
    'log_interval': 50,

    # Optimizer (default)
    'optimizer': {
        'type': 'AdamW',
        'lr': GLOBAL_LEARNING_RATE,
        'betas': (0.9, 0.999),
        'weight_decay': 0.01,
    },
    
    'scheduler': {
        'type': 'plateau',
        'warmup': 'linear',
        'warmup_iters': 1500,
        'warmup_ratio': 1e-6,
        'power': 1.0,
        'eta_min': 0.0,
        # ReduceLROnPlateau settings
        'mode': 'max',
        'factor': 0.5,
        'patience': 3,
        'min_lr': 1e-6,
    },
    
    'early_stopping': {
        'enabled': True,
        'monitor': 'mIoU', # or 'mIoU'
        'mode': 'max',     # 'min' for loss, 'max' for mIoU
        'patience': 5,
        'min_delta': 0.001,
        'stop_on_min_lr': True,
        'reset_on_lr_drop': True,
    },
}


INRIA_BASE_CONFIG = {
    **BASE_CONFIG,
    'num_classes': 2,
    'dataset': 'inria',
    'raw_data_root': 'data/inria/AerialImageDataset',
    'data_root': 'data/inria/AerialImageDataset_tiled',
    'crop_size': (224, 224),
}


# Swin Tiny configuration
SWIN_TINY_CONFIG = {
    **BASE_CONFIG,
    'model': {
        'encoder': 'swin_tiny',
        'decoder': 'upernet',
        'adapter': None,
        'train_encoder': True,
        'use_auxiliary_decoder': False,
        'name': 'swin_tiny',
        'pretrained': True,
        'pretrain_path': None,  # Will auto-download official ImageNet-22K Swin weights
        'encoder_kwargs': {
            'embed_dims': 96,
            'depths': [2, 2, 6, 2],
            'num_heads': [3, 6, 12, 24],
            'window_size': 7,
            'mlp_ratio': 4,
            'patch_size': 4,
            'drop_path_rate': 0.2,
            'use_checkpoint': False,
        },
        'decoder_kwargs': {
            'in_channels': [96, 192, 384, 768],
            'channels': 512,
            'dropout_ratio': 0.1,
        },
        'auxiliary_kwargs': {
            'in_channels': 384,
            'channels': 256,
            'num_convs': 1,
            'concat_input': False,
            'dropout_ratio': 0.1,
            'in_index': 2,
            'align_corners': False,
        },
    },
}


# Swin Small configuration
SWIN_SMALL_CONFIG = {
    **BASE_CONFIG,
    'model': {
        'encoder': 'swin_small',
        'decoder': 'upernet',
        'adapter': None,
        'train_encoder': True,
        'use_auxiliary_decoder': True,
        'name': 'swin_small',
        'pretrained': True,
        'pretrain_path': None,
        'encoder_kwargs': {
            'embed_dims': 96,
            'depths': [2, 2, 18, 2],
            'num_heads': [3, 6, 12, 24],
            'window_size': 7,
            'mlp_ratio': 4,
            'patch_size': 4,
            'drop_path_rate': 0.3,
            'use_checkpoint': False,
        },
        'decoder_kwargs': {
            'in_channels': [96, 192, 384, 768],
            'channels': 512,
            'dropout_ratio': 0.1,
        },
        'auxiliary_kwargs': {
            'in_channels': 384,
            'channels': 256,
            'num_convs': 1,
            'concat_input': False,
            'dropout_ratio': 0.1,
            'in_index': 2,
            'align_corners': False,
        },
    },
}


# Swin Base configuration
SWIN_BASE_CONFIG = {
    **BASE_CONFIG,
    'model': {
        'encoder': 'swin_base',
        'decoder': 'upernet',
        'adapter': None,
        'train_encoder': True,
        'use_auxiliary_decoder': False,
        'name': 'swin_base',
        'pretrained': True,
        'pretrain_path': None,
        'encoder_kwargs': {
            'embed_dims': 128,
            'depths': [2, 2, 18, 2],
            'num_heads': [4, 8, 16, 32],
            'window_size': 7,
            'mlp_ratio': 4,
            'patch_size': 4,
            'drop_path_rate': 0.3,
            'use_checkpoint': False,
        },
        'decoder_kwargs': {
            'in_channels': [128, 256, 512, 1024],
            'channels': 512,
            'dropout_ratio': 0.1,
        },
        'auxiliary_kwargs': {
            'in_channels': 512,
            'channels': 256,
            'num_convs': 1,
            'concat_input': False,
            'dropout_ratio': 0.1,
            'in_index': 2,
            'align_corners': False,
        },
    },
}


# Swin Large configuration
SWIN_LARGE_CONFIG = {
    **BASE_CONFIG,
    'model': {
        'encoder': 'swin_large',
        'decoder': 'upernet',
        'adapter': None,
        'train_encoder': True,
        'use_auxiliary_decoder': True,
        'name': 'swin_large',
        'pretrained': True,
        'pretrain_path': None,
        'encoder_kwargs': {
            'embed_dims': 192,
            'depths': [2, 2, 18, 2],
            'num_heads': [6, 12, 24, 48],
            'window_size': 7,
            'mlp_ratio': 4,
            'patch_size': 4,
            'drop_path_rate': 0.3,
            'use_checkpoint': False,
        },
        'decoder_kwargs': {
            'in_channels': [192, 384, 768, 1536],
            'channels': 512,
            'dropout_ratio': 0.1,
        },
        'auxiliary_kwargs': {
            'type': 'channorm_upernet',
            'in_channels': 768,
            'channels': 256,
            'num_convs': 1,
            'concat_input': False,
            'dropout_ratio': 0.1,
            'in_index': 2,
            'align_corners': False,
        },
    },
}


# Swin Base with Adapter and LPR Decoder configuration
SWIN_BASE_LPR_CONFIG = {
    **SWIN_BASE_CONFIG,
    'model': {
        **SWIN_BASE_CONFIG['model'],
        'adapter': 'swinb_lpr_adapter',
        'train_encoder': True,
        'use_auxiliary_decoder': False,
        'name': 'swin_base_lpr',
        'adapter_kwargs': {
            'in_channels': 1920,  # Sum of Swin Base channels: 128+256+512+1024
            'out_channels': 1024,
            'use_checkpoint': True,
        },
        'decoder': 'lpr',
        'decoder_kwargs': {
            # The adapter reduces the 4 feature maps into a single 256-channel tensor
            'in_channels': [1024],
            'lpr_kwargs': {
                'in_channels': 3,       # Image channels for the internal UNet
                'patch_size': 16,
                'hidden_dim': 384,
                'cnn_dim': 64,
                'use_checkpoint': True,
            },
        },
        'auxiliary_kwargs': {
            'type': 'upernet',
            'in_channels': 512,
            'channels': 256,
            'num_convs': 1,
            'concat_input': False,
            'dropout_ratio': 0.1,
            'in_index': 2,
            'align_corners': False,
        },
    },
}


# Swin Base with LPR High Resolution Decoder configuration
SWIN_BASE_LPR_HI_CONFIG = {
    **SWIN_BASE_CONFIG,
    'model': {
        **SWIN_BASE_CONFIG['model'],
        'adapter': None,
        'train_encoder': True,
        'use_auxiliary_decoder': False,
        'name': 'swin_base_lpr_hi',
        'decoder': 'lpr_hi',
        'decoder_kwargs': {
            # Process all multi-stage features directly from Swin Base
            'in_channels': [96, 192, 384, 768], #[128, 256, 512, 1024],
            'lpr_kwargs': {
                'in_channels': 3,       # Image channels for the internal UNet
                'hidden_dim': 256,
                'cnn_dim': 256,
                'use_checkpoint': False,
                'use_ppm': False,
                'attn_drop': 0, #0.1,
                'proj_drop': 0, #0.1,
                'drop_path_rate': 0, #0.1,
                'ppm_dropout': 0, #0.2,
                'spatial_dropout': 0, #0.2,
            }
        },
        'auxiliary_kwargs': {
            'type': 'upernet',
            'in_channels': 512,
            'channels': 256,
            'num_convs': 1,
            'concat_input': False,
            'dropout_ratio': 0.1,
            'in_index': 2,
            'align_corners': False,
        },
    },
}

# Swin Base with LPR High Resolution Decoder and no final PPM pooling
SWIN_BASE_LPR_HI_NOPOOL_CONFIG = {
    **SWIN_BASE_CONFIG,
    'model': {
        **SWIN_BASE_LPR_HI_CONFIG['model'],
        'name': 'swin_base_lpr_hi_nopool',
        'decoder_kwargs': {
            **SWIN_BASE_LPR_HI_CONFIG['model']['decoder_kwargs'],
            'lpr_kwargs': {
                **SWIN_BASE_LPR_HI_CONFIG['model']['decoder_kwargs']['lpr_kwargs'],
                'use_ppm': False,
            },
        },
    },
}


# Swin Base with UNet Decoder configuration
SWIN_BASE_UNET_CONFIG = {
    **SWIN_BASE_CONFIG,
    'model': {
        **SWIN_BASE_CONFIG['model'],
        'adapter': None,
        'train_encoder': True,
        'use_auxiliary_decoder': False,
        'name': 'swin_base_unet',
        'decoder': 'unet',
        'decoder_kwargs': {
            'in_channels': [128, 256, 512, 1024],
            'decoder_channels': [768, 512, 256], #[512, 256, 128],
            'num_convs': 3, #2,
            'dropout_ratio': 0.1,
            'align_corners': False,
            'output_scale': 4,
            'upsample_cfg': {
                'type': 'InterpConv',
                'scale_factor': 2,
                'mode': 'bilinear',
                'align_corners': False,
            },
        },
        'auxiliary_kwargs': {
            'type': 'upernet',
            'in_channels': 512,
            'channels': 256,
            'num_convs': 1,
            'concat_input': False,
            'dropout_ratio': 0.1,
            'in_index': 2,
            'align_corners': False,
        },
    },
}


def _build_swin_base_decoder_config(
    config_name: str,
    decoder_name: str,
    decoder_kwargs: dict,
    *,
    use_auxiliary_decoder: bool = False,
    auxiliary_kwargs: dict | None = None,
) -> dict:
    """Build a Swin Base variant for a specific decoder."""
    config = copy.deepcopy(SWIN_BASE_CONFIG)
    model = config['model']
    model['name'] = config_name
    model['decoder'] = decoder_name
    model['adapter'] = None
    model['train_encoder'] = True
    model['use_auxiliary_decoder'] = use_auxiliary_decoder
    model['decoder_kwargs'] = copy.deepcopy(decoder_kwargs)
    if auxiliary_kwargs is not None:
        model['auxiliary_kwargs'] = copy.deepcopy(auxiliary_kwargs)
    return config


SWIN_BASE_UPERNET_CONFIG = _build_swin_base_decoder_config(
    'swin_base_upernet',
    'upernet',
    {
        'in_channels': [128, 256, 512, 1024],
        'channels': 512,
        'dropout_ratio': 0.1,
    },
)

SWIN_BASE_FPN_CONFIG = _build_swin_base_decoder_config(
    'swin_base_fpn',
    'fpn',
    {
        'in_channels': [128, 256, 512, 1024],
        'channels': 512,
        'dropout_ratio': 0.1,
        'align_corners': False,
    },
)

SWIN_BASE_FCN_CONFIG = _build_swin_base_decoder_config(
    'swin_base_fcn',
    'fcn',
    {
        'in_channels': [128, 256, 512, 1024],
        'channels': 512,
        'num_convs': 2,
        'concat_input': False,
        'dropout_ratio': 0.1,
        'in_index': -1,
        'align_corners': False,
    },
)

SWIN_BASE_PSP_CONFIG = _build_swin_base_decoder_config(
    'swin_base_psp',
    'psp',
    {
        'in_channels': [128, 256, 512, 1024],
        'channels': 512,
        'dropout_ratio': 0.1,
        'align_corners': False,
    },
)

SWIN_BASE_DEEPLABV3_CONFIG = _build_swin_base_decoder_config(
    'swin_base_deeplabv3',
    'deeplabv3',
    {
        'in_channels': [128, 256, 512, 1024],
        'channels': 512,
        'dilations': (12, 24, 36),
        'dropout_ratio': 0.1,
        'align_corners': False,
    },
)

SWIN_BASE_DEEPLABV3PLUS_CONFIG = _build_swin_base_decoder_config(
    'swin_base_deeplabv3plus',
    'deeplabv3plus',
    {
        'in_channels': [128, 256, 512, 1024],
        'channels': 256,
        'dilations': (12, 24, 36),
        'dropout_ratio': 0.1,
        'align_corners': False,
        'low_level_index': 0,
        'low_level_channels': 48,
    },
)

SWIN_BASE_SEGFORMER_CONFIG = _build_swin_base_decoder_config(
    'swin_base_segformer',
    'segformer',
    {
        'in_channels': [128, 256, 512, 1024],
        'channels': 256,
        'dropout_ratio': 0.1,
        'align_corners': False,
    },
)

SWIN_BASE_OCR_CONFIG = _build_swin_base_decoder_config(
    'swin_base_ocr',
    'ocr',
    {
        'in_channels': [128, 256, 512, 1024],
        'channels': 512,
        'dropout_ratio': 0.1,
        'align_corners': False,
    },
)

SWIN_BASE_TCAD_CONFIG = _build_swin_base_decoder_config(
    'swin_base_tcad',
    'tcad',
    {
        'in_channels': [128, 256, 512, 1024],
        'decoder_kwargs': {
            'hidden_dim': 256,
            'use_checkpoint': False,
            'use_ppm': False,
            'attn_drop': 0, #0.1,
            'proj_drop': 0, #0.1,
            'drop_path_rate': 0, #0.1,
            'ppm_dropout': 0, #0.2,
            'spatial_dropout': 0, #0.2,
        },
    },
)


# Configuration dictionary for easy access
CONFIG = {
    'swin_tiny': SWIN_TINY_CONFIG,
    'swin_small': SWIN_SMALL_CONFIG,
    'swin_base': SWIN_BASE_CONFIG,
    'swin_base_upernet': SWIN_BASE_UPERNET_CONFIG,
    'swin_base_fpn': SWIN_BASE_FPN_CONFIG,
    'swin_base_fcn': SWIN_BASE_FCN_CONFIG,
    'swin_base_psp': SWIN_BASE_PSP_CONFIG,
    'swin_base_deeplabv3': SWIN_BASE_DEEPLABV3_CONFIG,
    'swin_base_deeplabv3plus': SWIN_BASE_DEEPLABV3PLUS_CONFIG,
    'swin_base_segformer': SWIN_BASE_SEGFORMER_CONFIG,
    'swin_base_ocr': SWIN_BASE_OCR_CONFIG,
    'swin_base_tcad': SWIN_BASE_TCAD_CONFIG,
    'swin_base_unet': SWIN_BASE_UNET_CONFIG,
    'swin_large': SWIN_LARGE_CONFIG,
    'swin_base_lpr': SWIN_BASE_LPR_CONFIG,
    'swin_base_lpr_hi': SWIN_BASE_LPR_HI_CONFIG,
    'swin_base_lpr_hi_nopool': SWIN_BASE_LPR_HI_NOPOOL_CONFIG,
}


DATASET_PRESETS = {
    'ade20k': {
        'dataset': 'ade20k',
        'data_root': 'data/ade/ADEChallengeData2016',
        'crop_size': (512, 512),
    },
    'inria': {
        'dataset': 'inria',
        'num_classes': 2,
        'raw_data_root': 'data/inria/AerialImageDataset',
        'data_root': 'data/inria/AerialImageDataset_tiled',
        'crop_size': (224, 224),
    },
}


def build_config(config_name: str, dataset_name: str | None = None) -> dict:
    """Build a runtime config by combining a backbone config with a dataset preset."""
    if config_name not in CONFIG:
        raise KeyError(f'Unknown config: {config_name}')

    config = copy.deepcopy(CONFIG[config_name])
    if dataset_name is None:
        return config

    dataset_key = dataset_name.lower()
    if dataset_key not in DATASET_PRESETS:
        raise KeyError(f'Unknown dataset: {dataset_name}')

    return _deep_merge_dicts(config, DATASET_PRESETS[dataset_key])


# Model variant details (for reference)
MODEL_DETAILS = {
    'swin_tiny': {
        'embed_dims': 96,
        'depths': [2, 2, 6, 2],
        'params': '60M',
        'flops': '945G',
    },
    'swin_small': {
        'embed_dims': 96,
        'depths': [2, 2, 18, 2],
        'params': '81M',
        'flops': '1038G',
    },
    'swin_base': {
        'embed_dims': 128,
        'depths': [2, 2, 18, 2],
        'params': '121M',
        'flops': '1841G',
    },
    'swin_base_upernet': {
        'embed_dims': 128,
        'depths': [2, 2, 18, 2],
        'params': '121M',
        'flops': '1841G',
    },
    'swin_base_fpn': {
        'embed_dims': 128,
        'depths': [2, 2, 18, 2],
        'params': '121M',
        'flops': '1841G',
    },
    'swin_base_fcn': {
        'embed_dims': 128,
        'depths': [2, 2, 18, 2],
        'params': '121M',
        'flops': '1841G',
    },
    'swin_base_psp': {
        'embed_dims': 128,
        'depths': [2, 2, 18, 2],
        'params': '121M',
        'flops': '1841G',
    },
    'swin_base_deeplabv3': {
        'embed_dims': 128,
        'depths': [2, 2, 18, 2],
        'params': '121M',
        'flops': '1841G',
    },
    'swin_base_deeplabv3plus': {
        'embed_dims': 128,
        'depths': [2, 2, 18, 2],
        'params': '121M',
        'flops': '1841G',
    },
    'swin_base_segformer': {
        'embed_dims': 128,
        'depths': [2, 2, 18, 2],
        'params': '121M',
        'flops': '1841G',
    },
    'swin_base_ocr': {
        'embed_dims': 128,
        'depths': [2, 2, 18, 2],
        'params': '121M',
        'flops': '1841G',
    },
    'swin_large': {
        'embed_dims': 192,
        'depths': [2, 2, 18, 2],
        'params': '234M',
        'flops': '3230G',
    },
    'swin_base_unet': {
        'embed_dims': 128,
        'depths': [2, 2, 18, 2],
        'params': '121M',
        'flops': '1841G',
    },
}


def print_config(config_name: str):
    """Print configuration for a specific model."""
    if config_name not in CONFIG:
        print(f"Unknown config: {config_name}")
        return
    
    config = CONFIG[config_name]
    details = MODEL_DETAILS.get(config_name, {})
    
    print(f"\n{'='*60}")
    print(f"Configuration: {config_name}")
    print(f"{'='*60}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['optimizer']['lr']}")
    print(f"Max iterations: {config['train_cfg']['max_iters']}")
    print(f"Validation interval: {config['train_cfg']['val_interval']} iters")
    if config.get('dataset') == 'inria':
        print(f"Raw data root: {config.get('raw_data_root', 'N/A')}")
        print(f"Prepared data root: {config.get('data_root', 'N/A')}")
    print(f"\nModel architecture:")
    print(f"  encoder: {config['model']['encoder']}")
    print(f"  encoder_trainable: {config['model'].get('train_encoder', True)}")
    print(f"  aux_decoder_enabled: {config['model'].get('use_auxiliary_decoder', True)}")
    for key, val in config['model'].get('encoder_kwargs', {}).items():
        print(f"  encoder.{key}: {val}")
    print(f"  decoder: {config['model']['decoder']}")
    for key, val in config['model'].get('decoder_kwargs', {}).items():
        print(f"  decoder.{key}: {val}")
    if config['model'].get('adapter'):
        print(f"  adapter: {config['model']['adapter']}")
        for key, val in config['model'].get('adapter_kwargs', {}).items():
            print(f"  adapter.{key}: {val}")
    if config['model'].get('use_auxiliary_decoder', True):
        for key, val in config['model'].get('auxiliary_kwargs', {}).items():
            print(f"  auxiliary.{key}: {val}")
    print(f"\nModel params: {details.get('params', 'N/A')}")
    print(f"Model FLOPs: {details.get('flops', 'N/A')}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    # Print all configurations
    for config_name in CONFIG.keys():
        print_config(config_name)
