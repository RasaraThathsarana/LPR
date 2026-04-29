"""FPN decoder package."""

from .model import FPNHead

__all__ = [
    'FPNHead',
    'build_decoder',
]


def build_decoder(**kwargs):
    return FPNHead(**kwargs)
