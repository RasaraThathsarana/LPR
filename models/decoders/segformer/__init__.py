"""SegFormer decoder package."""

from .model import SegFormerHead

__all__ = [
    'SegFormerHead',
    'build_decoder',
]


def build_decoder(**kwargs):
    return SegFormerHead(**kwargs)
