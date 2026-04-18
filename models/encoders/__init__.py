"""Encoder implementations for segmentation models."""

import importlib
import pkgutil
from typing import Any, Dict, List

__all__ = [
    'build_encoder',
    'list_encoders',
    'get_encoder_config',
]


def _import_encoder_module(name: str):
    try:
        return importlib.import_module(f'models.encoders.{name}')
    except ImportError as error:
        raise ValueError(f'Unknown encoder: {name}') from error


def build_encoder(name: str, **kwargs: Any):
    module = _import_encoder_module(name.lower())
    if hasattr(module, 'build_encoder'):
        return module.build_encoder(**kwargs)
    raise AttributeError(f'Encoder module {name} does not expose build_encoder')


def list_encoders() -> List[str]:
    return [module.name for module in pkgutil.iter_modules(__path__)]


def get_encoder_config(name: str) -> Dict[str, Any]:
    module = _import_encoder_module(name.lower())
    if hasattr(module, 'DEFAULT_CONFIG'):
        return getattr(module, 'DEFAULT_CONFIG')
    raise AttributeError(f'Encoder module {name} does not expose DEFAULT_CONFIG')
