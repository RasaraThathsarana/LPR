"""Swin encoder package."""

from .config import DEFAULT_CONFIG
from .model import SwinEncoder

__all__ = [
    'SwinEncoder',
    'build_encoder',
    'DEFAULT_CONFIG',
]


def build_encoder(**kwargs):
    config = {**DEFAULT_CONFIG, **kwargs}
    return SwinEncoder(**config)
