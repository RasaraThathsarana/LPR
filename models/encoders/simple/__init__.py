"""Simple encoder package."""

from .config import DEFAULT_CONFIG
from .model import SimpleEncoder

__all__ = [
    'SimpleEncoder',
    'build_encoder',
    'DEFAULT_CONFIG',
]


def build_encoder(**kwargs):
    config = {**DEFAULT_CONFIG, **kwargs}
    return SimpleEncoder(**config)
