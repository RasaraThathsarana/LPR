"""Simple decoder package."""

from .config import DEFAULT_CONFIG
from .model import SimpleDecoder

__all__ = [
    'SimpleDecoder',
    'build_decoder',
    'DEFAULT_CONFIG',
]


def build_decoder(**kwargs):
    config = {**DEFAULT_CONFIG, **kwargs}
    return SimpleDecoder(**config)
