"""LPR Decoder Package."""

from .model import LPRDecoder

def build_decoder(**kwargs):
    """Builds the LPR Decoder."""
    return LPRDecoder(**kwargs)