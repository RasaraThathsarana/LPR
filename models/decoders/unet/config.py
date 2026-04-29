"""Default configuration for the UNet decoder."""

DEFAULT_CONFIG = {
    'in_channels': [128, 256, 512, 1024],
    'decoder_channels': [512, 256, 128],
    'num_classes': 150,
    'num_convs': 2,
    'dropout_ratio': 0.1,
    'align_corners': False,
    'output_scale': 4,
    'upsample_cfg': {
        'type': 'InterpConv',
        'scale_factor': 2,
        'mode': 'bilinear',
        'align_corners': False,
    },
}
