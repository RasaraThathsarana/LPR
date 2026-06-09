"""Default configuration for the TCAD decoder."""

DEFAULT_CONFIG = {
    'in_channels': [128, 256, 512, 1024],
    'decoder_kwargs': {
        'hidden_dim': 256,
        'use_checkpoint': False,
        'use_ppm': True,
        'attn_drop': 0.1,
        'proj_drop': 0.1,
        'drop_path_rate': 0.1,
        'ppm_dropout': 0.2,
        'spatial_dropout': 0.2,
    },
}
