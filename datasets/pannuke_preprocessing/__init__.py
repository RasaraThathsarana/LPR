"""PanNuke preprocessing package.

Provides tools to convert PanNuke instance annotations to semantic segmentation
maps and a dataset class compatible with the project's dataset interface.
"""

from .pannuke_dataset import PanNukeDataset
from .preprocessing import (
    create_semantic_masks_from_instances,
    instances_to_semantic,
    build_pipeline,
)
from .dataloader import create_train_loader, create_val_loader
from .preprocessing_config import (
    DATASET_TYPE,
    RAW_DATA_ROOT,
    DATA_ROOT,
    CROP_SIZE,
    TRAIN_PIPELINE,
    VAL_PIPELINE,
)
from .download import download_instructions, ensure_data_available

__all__ = [
    "PanNukeDataset",
    "create_semantic_masks_from_instances",
    "instances_to_semantic",
    "build_pipeline",
    "create_train_loader",
    "create_val_loader",
    "DATASET_TYPE",
    "RAW_DATA_ROOT",
    "DATA_ROOT",
    "CROP_SIZE",
    "TRAIN_PIPELINE",
    "VAL_PIPELINE",
    "download_instructions",
    "ensure_data_available",
]
