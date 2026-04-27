"""Preprocessing pipeline configuration for the Inria dataset."""

DATASET_TYPE = 'InriaAerialImageDataset'
DATA_ROOT = 'data/inria/AerialImageDataset'
CROP_SIZE = (224, 224)
TILE_SIZE = 224
LARGE_IMAGE_THRESHOLD = 512

TRAIN_PIPELINE = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375), to_rgb=False),
    dict(type='PackSegInputs'),
]

VAL_PIPELINE = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Normalize', mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375), to_rgb=False),
    dict(type='PackSegInputs'),
]

TRAIN_BATCH_SIZE = 8
VAL_BATCH_SIZE = 1
NUM_WORKERS = 4
