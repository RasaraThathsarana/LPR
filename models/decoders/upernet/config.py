"""Default configuration for the UPerNet decoder."""

DEFAULT_CONFIG = {
    'in_channels': [96, 192, 384, 768],
    'channels': 512,
    'num_classes': 150,
    'dropout_ratio': 0.1,
}
