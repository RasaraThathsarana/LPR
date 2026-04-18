"""Simple adapter package."""

from .config import DEFAULT_CONFIG
from .model import SimpleAdapter

__all__ = [
    'SimpleAdapter',
    'build_adapter',
    'DEFAULT_CONFIG',
]


def build_adapter(**kwargs):
    config = {**DEFAULT_CONFIG, **kwargs}
    return SimpleAdapter(**config)
