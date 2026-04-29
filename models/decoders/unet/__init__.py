"""UNet decoder package."""

from .config import DEFAULT_CONFIG
from .model import UNetDecoder

__all__ = [
    'UNetDecoder',
    'build_decoder',
    'DEFAULT_CONFIG',
]


def build_decoder(**kwargs):
    config = {**DEFAULT_CONFIG, **kwargs}
    return UNetDecoder(**config)

