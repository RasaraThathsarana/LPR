"""Decoder implementations for segmentation models."""

import importlib
import pkgutil
from typing import Any, Dict, List

__all__ = [
    'build_decoder',
    'list_decoders',
    'get_decoder_config',
]


def _import_decoder_module(name: str):
    try:
        return importlib.import_module(f'models.decoders.{name}')
    except ImportError as error:
        raise ValueError(f'Unknown decoder: {name}') from error


def build_decoder(name: str, **kwargs: Any):
    module = _import_decoder_module(name.lower())
    if hasattr(module, 'build_decoder'):
        return module.build_decoder(**kwargs)
    raise AttributeError(f'Decoder module {name} does not expose build_decoder')


def list_decoders() -> List[str]:
    return [module.name for module in pkgutil.iter_modules(__path__)]


def get_decoder_config(name: str) -> Dict[str, Any]:
    module = _import_decoder_module(name.lower())
    if hasattr(module, 'DEFAULT_CONFIG'):
        return getattr(module, 'DEFAULT_CONFIG')
    raise AttributeError(f'Decoder module {name} does not expose DEFAULT_CONFIG')
