"""TCAD decoder package."""

from .config import DEFAULT_CONFIG
from .model import TopDownDecoder

__all__ = [
    'TopDownDecoder',
    'build_decoder',
    'DEFAULT_CONFIG',
]


def build_decoder(**kwargs):
    config = {**DEFAULT_CONFIG, **kwargs}
    return TopDownDecoder(**config)
