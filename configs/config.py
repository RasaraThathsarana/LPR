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
    
    # Training settings (matches MMSeg schedule_160k.py)
    'train_cfg': {
        'max_iters': 1000, #160000,
        'val_interval': 20, #16000,  # Validate every 16k iterations
    },
    
    # Data loading
    'num_workers': 4,
    'pin_memory': True,
    
    # Logging
    'log_interval': 50,
    'num_epochs': 160,  # Roughly 160k / (20k samples / batch_size)
    'val_interval': 1,  # Validate every epoch
    
    # Optimizer (default)
    'optimizer': {
        'type': 'SGD',
        'lr': 0.01,
        'momentum': 0.9,
        'weight_decay': 0.0005,
    },
    
    'scheduler': {
        'type': 'poly',
        'power': 0.9,
        'eta_min': 1e-4,
    },
}


# Swin Tiny configuration
SWIN_TINY_CONFIG = {
    **BASE_CONFIG,
    'batch_size': 4,  # 2 per GPU * 4 GPUs (adjust based on your setup)
    'model': {
        'encoder': 'swin_tiny',
        'decoder': 'upernet',
        'adapter': None,
        'name': 'swin_tiny',
        'pretrained': True,
        'pretrain_path': None,  # Will auto-download from MMSeg
        'encoder_kwargs': {
            'embed_dims': 96,
            'depths': [2, 2, 6, 2],
            'num_heads': [3, 6, 12, 24],
            'window_size': 7,
            'mlp_ratio': 4,
            'patch_size': 4,
            'drop_path_rate': 0.2,
        },
        'decoder_kwargs': {
            'in_channels': [96, 192, 384, 768],
            'channels': 512,
            'dropout_ratio': 0.1,
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
        },
        'decoder_kwargs': {
            'in_channels': [96, 192, 384, 768],
            'channels': 512,
            'dropout_ratio': 0.1,
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
    'batch_size': 8,  # Might need to reduce batch size for base model
    'model': {
        'encoder': 'swin_base',
        'decoder': 'upernet',
        'adapter': None,
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
        },
        'decoder_kwargs': {
            'in_channels': [128, 256, 512, 1024],
            'channels': 512,
            'dropout_ratio': 0.1,
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
        },
        'decoder_kwargs': {
            'in_channels': [192, 384, 768, 1536],
            'channels': 512,
            'dropout_ratio': 0.1,
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
        'params': '29M',
        'flops': '39G',
    },
    'swin_small': {
        'embed_dims': 96,
        'depths': [2, 2, 18, 2],
        'params': '50M',
        'flops': '83G',
    },
    'swin_base': {
        'embed_dims': 128,
        'depths': [2, 2, 18, 2],
        'params': '88M',
        'flops': '140G',
    },
    'swin_large': {
        'embed_dims': 192,
        'depths': [2, 2, 18, 2],
        'params': '197M',
        'flops': '282G',
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
    for key, val in config['model']['backbone'].items():
        print(f"  {key}: {val}")
    print(f"\nModel params: {details.get('params', 'N/A')}")
    print(f"Model FLOPs: {details.get('flops', 'N/A')}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    # Print all configurations
    for config_name in CONFIG.keys():
        print_config(config_name)
