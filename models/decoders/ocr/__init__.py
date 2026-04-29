"""OCR decoder package."""

from .model import OCRHead

__all__ = [
    'OCRHead',
    'build_decoder',
]


def build_decoder(**kwargs):
    return OCRHead(**kwargs)
