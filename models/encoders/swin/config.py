"""Default configuration for the Swin encoder."""

DEFAULT_CONFIG = {
    'in_channels': 3,
    'embed_dims': 96,
    'depths': [2, 2, 6, 2],
    'num_heads': [3, 6, 12, 24],
    'window_size': 7,
    'mlp_ratio': 4.0,
    'patch_size': 4,
    'drop_path_rate': 0.2,
}
