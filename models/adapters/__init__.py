"""Adapter implementations for segmentation models."""

import importlib
import pkgutil
from typing import Any, Dict, List

__all__ = [
    'build_adapter',
    'list_adapters',
    'get_adapter_config',
]


def _import_adapter_module(name: str):
    try:
        return importlib.import_module(f'models.adapters.{name}')
    except ImportError as error:
        raise ValueError(f'Unknown adapter: {name}') from error


def build_adapter(name: str, **kwargs: Any):
    module = _import_adapter_module(name.lower())
    if hasattr(module, 'build_adapter'):
        return module.build_adapter(**kwargs)
    raise AttributeError(f'Adapter module {name} does not expose build_adapter')


def list_adapters() -> List[str]:
    return [module.name for module in pkgutil.iter_modules(__path__)]


def get_adapter_config(name: str) -> Dict[str, Any]:
    module = _import_adapter_module(name.lower())
    if hasattr(module, 'DEFAULT_CONFIG'):
        return getattr(module, 'DEFAULT_CONFIG')
    raise AttributeError(f'Adapter module {name} does not expose DEFAULT_CONFIG')
