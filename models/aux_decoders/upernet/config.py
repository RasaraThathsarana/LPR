"""Default configuration for the UPerNet auxiliary decoder."""

DEFAULT_CONFIG = {
    'in_channels': 384,
    'channels': 256,
    'num_convs': 1,
    'concat_input': False,
    'num_classes': 150,
    'dropout_ratio': 0.1,
    'in_index': 2,
    'align_corners': False,
}
