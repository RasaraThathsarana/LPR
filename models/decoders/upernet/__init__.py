"""UPerNet decoder package."""

from .config import DEFAULT_CONFIG
from .model import UPerNetDecoder

__all__ = [
    'UPerNetDecoder',
    'build_decoder',
    'DEFAULT_CONFIG',
]


def build_decoder(**kwargs):
    config = {**DEFAULT_CONFIG, **kwargs}
    return UPerNetDecoder(**config)
