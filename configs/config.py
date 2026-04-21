"""
Training configurations for Swin UPerNet models.

These configurations replicate MMSegmentation's settings for:
- Swin Tiny
- Swin Small
- Swin Base
- Swin Large
"""

# Base configuration common to all variants
BASE_CONFIG = {
    'num_classes': 150,
    'data_root': 'data/ade/ADEChallengeData2016',
    'crop_size': (512, 512),
    'data_preprocessor': {
        'mean': [123.675, 116.28, 103.53],
        'std': [58.395, 57.12, 57.375],
        'to_rgb': False,
    },
    
    # Training settings (matches MMSeg schedule_160k.py)
    'train_cfg': {
        'max_iters': 100000,
        'val_interval': 2527,
    },
    
    # Data loading
    'num_workers': 4,
    'pin_memory': True,
    
    # Logging
    'log_interval': 50,

    # Optimizer (default)
    'optimizer': {
        'type': 'SGD',
        'lr': 0.01,
        'momentum': 0.9,
        'weight_decay': 0.0005,
    },
    
    'scheduler': {
        'type': 'poly',
        'warmup': 'linear',
        'warmup_iters': 1500,
        'warmup_ratio': 1e-6,
        'power': 1.0,
        'eta_min': 0.0,
    },
}


# Swin Tiny configuration
SWIN_TINY_CONFIG = {
    **BASE_CONFIG,
    'batch_size': 2,  # 2 per GPU * 4 GPUs (adjust based on your setup)
    'model': {
        'encoder': 'swin_tiny',
        'decoder': 'upernet',
        'adapter': None,
        'use_auxiliary_decoder': True,
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
    'optimizer': {
        'type': 'AdamW',
        'lr': 6e-5,
        'betas': (0.9, 0.999),
        'weight_decay': 0.01,
    },
}


# Swin Small configuration
SWIN_SMALL_CONFIG = {
    **BASE_CONFIG,
    'batch_size': 8,
    'model': {
        'encoder': 'swin_small',
        'decoder': 'upernet',
        'adapter': None,
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
    'optimizer': {
        'type': 'AdamW',
        'lr': 6e-5,
        'betas': (0.9, 0.999),
        'weight_decay': 0.01,
    },
}


# Swin Base configuration
SWIN_BASE_CONFIG = {
    **BASE_CONFIG,
    'batch_size': 8,
    'model': {
        'encoder': 'swin_base',
        'decoder': 'upernet',
        'adapter': None,
        'use_auxiliary_decoder': True,
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
            'use_checkpoint': True,
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
    'optimizer': {
        'type': 'AdamW',
        'lr': 6e-5,
        'betas': (0.9, 0.999),
        'weight_decay': 0.01,
    },
}


# Swin Large configuration
SWIN_LARGE_CONFIG = {
    **BASE_CONFIG,
    'batch_size': 2,  # Reduce batch size for large model
    'model': {
        'encoder': 'swin_large',
        'decoder': 'upernet',
        'adapter': None,
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
            'in_channels': 768,
            'channels': 256,
            'num_convs': 1,
            'concat_input': False,
            'dropout_ratio': 0.1,
            'in_index': 2,
            'align_corners': False,
        },
    },
    'optimizer': {
        'type': 'AdamW',
        'lr': 6e-5,
        'betas': (0.9, 0.999),
        'weight_decay': 0.01,
    },
}


# Configuration dictionary for easy access
CONFIG = {
    'swin_tiny': SWIN_TINY_CONFIG,
    'swin_small': SWIN_SMALL_CONFIG,
    'swin_base': SWIN_BASE_CONFIG,
    'swin_large': SWIN_LARGE_CONFIG,
}


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
    'swin_large': {
        'embed_dims': 192,
        'depths': [2, 2, 18, 2],
        'params': '234M',
        'flops': '3230G',
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
    print(f"\nModel architecture:")
    print(f"  encoder: {config['model']['encoder']}")
    print(f"  aux_decoder_enabled: {config['model'].get('use_auxiliary_decoder', True)}")
    for key, val in config['model'].get('encoder_kwargs', {}).items():
        print(f"  encoder.{key}: {val}")
    print(f"  decoder: {config['model']['decoder']}")
    for key, val in config['model'].get('decoder_kwargs', {}).items():
        print(f"  decoder.{key}: {val}")
    print(f"\nModel params: {details.get('params', 'N/A')}")
    print(f"Model FLOPs: {details.get('flops', 'N/A')}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    # Print all configurations
    for config_name in CONFIG.keys():
        print_config(config_name)
