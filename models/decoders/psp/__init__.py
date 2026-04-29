"""PSP decoder package."""

from .model import PSPHead

__all__ = [
    'PSPHead',
    'build_decoder',
]


def build_decoder(**kwargs):
    return PSPHead(**kwargs)
