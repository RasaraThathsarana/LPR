"""Configuration defaults for PanNuke preprocessing."""

DEFAULT_CLASS_MAPPING = {
    # PanNuke classes are commonly numbered 1..5. Map them to semantic ids.
    # 1: neoplastic, 2: non-neoplastic epithelial, 3: inflammatory,
    # 4: connective, 5: dead
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
}

# File name patterns
INSTANCE_MASK_SUFFIXES = ('.npy', '.npz', '.png', '.tif', '.tiff')

# Pipeline settings
DATASET_TYPE = 'PanNukeDataset'
RAW_DATA_ROOT = 'data/pannuke_raw'
DATA_ROOT = 'data/pannuke'
CROP_SIZE = (256, 256)

TRAIN_PIPELINE = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]

VAL_PIPELINE = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]

TRAIN_BATCH_SIZE = 8
VAL_BATCH_SIZE = 1
NUM_WORKERS = 4
