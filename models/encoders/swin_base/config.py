"""Default configuration for the Swin Base encoder."""

DEFAULT_CONFIG = {
    'in_channels': 3,
    'embed_dims': 128,
    'depths': [2, 2, 18, 2],
    'num_heads': [4, 8, 16, 32],
    'window_size': 7,
    'mlp_ratio': 4.0,
    'patch_size': 4,
    'drop_path_rate': 0.3,
}
