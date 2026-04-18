"""Swin Base encoder package."""

from ..swin.model import SwinEncoder
from .config import DEFAULT_CONFIG

__all__ = [
    'SwinEncoder',
    'build_encoder',
    'DEFAULT_CONFIG',
]


def build_encoder(**kwargs):
    config = {**DEFAULT_CONFIG, **kwargs}
    return SwinEncoder(**config)
