"""FCN decoder package."""

from .model import FCNHead

__all__ = [
    'FCNHead',
    'build_decoder',
]


def build_decoder(**kwargs):
    return FCNHead(**kwargs)
