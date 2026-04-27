"""LPR-Hi decoder package."""

from .model import LPRHiDecoder


def build_decoder(**kwargs):
    """Build the LPR-Hi decoder."""
    return LPRHiDecoder(**kwargs)


__all__ = ['LPRHiDecoder', 'build_decoder']
