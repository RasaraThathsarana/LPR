"""Swin Base to LPR Adapter."""

from .model import LPRAdapter

def build_adapter(**kwargs):
    """Builds the LPR Adapter."""
    return LPRAdapter(**kwargs)