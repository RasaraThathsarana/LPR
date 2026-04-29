"""DeepLabV3 decoder package."""

from .model import DeepLabV3Head

__all__ = [
    'DeepLabV3Head',
    'build_decoder',
]


def build_decoder(**kwargs):
    return DeepLabV3Head(**kwargs)
