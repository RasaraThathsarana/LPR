"""Auxiliary decoder packages."""

import importlib
import pkgutil
from typing import Any, Dict, List

__all__ = [
    'build_auxiliary_head',
    'list_aux_decoders',
    'get_aux_decoder_config',
]


def _import_aux_decoder_module(name: str):
    try:
        return importlib.import_module(f'models.aux_decoders.{name}')
    except ImportError as error:
        raise ValueError(f'Unknown auxiliary decoder: {name}') from error


def build_auxiliary_head(name: str, **kwargs: Any):
    module = _import_aux_decoder_module(name.lower())
    if hasattr(module, 'build_auxiliary_head'):
        return module.build_auxiliary_head(**kwargs)
    raise AttributeError(f'Auxiliary decoder module {name} does not expose build_auxiliary_head')


def list_aux_decoders() -> List[str]:
    return [module.name for module in pkgutil.iter_modules(__path__)]


def get_aux_decoder_config(name: str) -> Dict[str, Any]:
    module = _import_aux_decoder_module(name.lower())
    if hasattr(module, 'DEFAULT_CONFIG'):
        return getattr(module, 'DEFAULT_CONFIG')
    raise AttributeError(f'Auxiliary decoder module {name} does not expose DEFAULT_CONFIG')
