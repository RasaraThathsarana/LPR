"""Identity adapter package."""

from .config import DEFAULT_CONFIG
from .model import IdentityAdapter

__all__ = [
    'IdentityAdapter',
    'build_adapter',
    'DEFAULT_CONFIG',
]


def build_adapter(**kwargs):
    config = {**DEFAULT_CONFIG, **kwargs}
    return IdentityAdapter(**config)
