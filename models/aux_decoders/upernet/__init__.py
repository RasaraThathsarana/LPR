"""UPerNet auxiliary decoder package."""

from .config import DEFAULT_CONFIG
from .model import AuxiliaryFCNHead

__all__ = [
    'AuxiliaryFCNHead',
    'build_auxiliary_head',
    'DEFAULT_CONFIG',
]


def build_auxiliary_head(**kwargs):
    aux_config = {
        'in_channels': DEFAULT_CONFIG['in_channels'],
        'channels': 256,
        'num_convs': 1,
        'concat_input': False,
        'num_classes': DEFAULT_CONFIG['num_classes'],
        'dropout_ratio': DEFAULT_CONFIG['dropout_ratio'],
        'in_index': 2,
        'align_corners': False,
    }
    aux_config.update(kwargs)
    return AuxiliaryFCNHead(**aux_config)
