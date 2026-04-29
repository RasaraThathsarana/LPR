"""DeepLabV3+ decoder package."""

from .model import DeepLabV3PlusHead

__all__ = [
    'DeepLabV3PlusHead',
    'build_decoder',
]


def build_decoder(**kwargs):
    return DeepLabV3PlusHead(**kwargs)
