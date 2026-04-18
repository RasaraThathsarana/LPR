"""
ADE20K Preprocessing Module

A standalone preprocessing module that replicates MMSegmentation's ADE20K preprocessing.
"""

from .ade20k_dataset import ADE20KDataset
from .preprocessing import (
    Compose,
    LoadImageFromFile,
    LoadAnnotations,
    RandomResize,
    Resize,
    RandomCrop,
    RandomFlip,
    PhotoMetricDistortion,
    Normalize,
    PackSegInputs,
    build_pipeline,
)
from .dataloader import ADE20KDataLoader, create_train_loader, create_val_loader
from .download import ensure_ade20k_dataset

__all__ = [
    'ADE20KDataset',
    'Compose',
    'LoadImageFromFile',
    'LoadAnnotations',
    'RandomResize',
    'Resize',
    'RandomCrop',
    'RandomFlip',
    'PhotoMetricDistortion',
    'Normalize',
    'PackSegInputs',
    'build_pipeline',
    'ADE20KDataLoader',
    'create_train_loader',
    'create_val_loader',
    'ensure_ade20k_dataset',
]

__version__ = '1.0.0'
