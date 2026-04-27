"""Dataset utilities and preprocessing for Swin UPerNet."""

from .ade20k_preprocessing import ADE20KDataset
from .inria_preprocessing import InriaAerialImageDataset


def create_train_loader(
    data_root,
    pipeline_config,
    batch_size: int = 4,
    shuffle: bool = True,
    dataset_name: str = 'ade20k',
    **kwargs,
):
    dataset_name = (dataset_name or 'ade20k').lower()
    if dataset_name == 'ade20k':
        from .ade20k_preprocessing import create_train_loader as _create_train_loader
        return _create_train_loader(data_root, pipeline_config, batch_size=batch_size, shuffle=shuffle, **kwargs)
    if dataset_name == 'inria':
        from .inria_preprocessing import create_train_loader as _create_train_loader
        return _create_train_loader(data_root, pipeline_config, batch_size=batch_size, shuffle=shuffle, **kwargs)
    raise ValueError(f'Unknown dataset: {dataset_name}')


def create_val_loader(
    data_root,
    pipeline_config,
    batch_size: int = 1,
    dataset_name: str = 'ade20k',
    **kwargs,
):
    dataset_name = (dataset_name or 'ade20k').lower()
    if dataset_name == 'ade20k':
        from .ade20k_preprocessing import create_val_loader as _create_val_loader
        return _create_val_loader(data_root, pipeline_config, batch_size=batch_size, **kwargs)
    if dataset_name == 'inria':
        from .inria_preprocessing import create_val_loader as _create_val_loader
        return _create_val_loader(data_root, pipeline_config, batch_size=batch_size, **kwargs)
    raise ValueError(f'Unknown dataset: {dataset_name}')


__all__ = [
    'ADE20KDataset',
    'InriaAerialImageDataset',
    'create_train_loader',
    'create_val_loader',
]
