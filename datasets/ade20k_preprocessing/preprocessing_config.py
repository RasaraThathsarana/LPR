"""
ADE20K Preprocessing Pipeline Configuration.

This configuration keeps ADE20K preprocessing configurable from the runtime
config while preserving the same transform structure.
"""

from copy import deepcopy

# Dataset settings
DATASET_TYPE = 'ADE20KDataset'
DATA_ROOT = 'data/ade/ADEChallengeData2016'
CROP_SIZE = (224, 224)

# Training pipeline
# These transforms are applied in order during training
TRAIN_PIPELINE = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='RandomResize',
        scale=(2048, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True
    ),
    dict(type='RandomCrop', crop_size=CROP_SIZE, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]


def get_train_pipeline(crop_size: tuple[int, int] | None = None):
    """Return a copy of the training pipeline with the requested crop size."""
    pipeline = deepcopy(TRAIN_PIPELINE)
    target_crop_size = crop_size or CROP_SIZE
    for transform in pipeline:
        if transform.get('type') == 'RandomCrop':
            transform['crop_size'] = target_crop_size
            break
    return pipeline


def get_val_pipeline():
    """Return a copy of the validation pipeline."""
    return deepcopy(VAL_PIPELINE)

# Validation/Test pipeline
# Note: No augmentation, only resizing and packing
# Lower the validation resize scale to reduce GPU memory use during full-image evaluation,
# especially for high-resolution decoders like lpr_hi.
VAL_PIPELINE = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(768, 384), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
]

# Data loader settings
TRAIN_BATCH_SIZE = 4
VAL_BATCH_SIZE = 1
NUM_WORKERS = 4

# Statistics for normalization (applied inside the model wrapper)
IMG_NORM_CFG = dict(
    mean=[123.675, 116.28, 103.53],  # ImageNet mean in RGB
    std=[58.395, 57.12, 57.375],     # ImageNet std in RGB
    to_rgb=False
)

# Preprocessing summary:
# =====================
# 1. Load RGB image (.jpg) and segmentation map (.png)
# 2. Reduce zero labels (shift all labels -1, background becomes 255)
# 3. Random resize: scale to (2048, 512) with ratio 0.5-2.0 keeping aspect ratio
# 4. Random crop: crop to 224x224 with max 75% of single class
# 5. Random horizontal flip: 50% probability
# 6. Photometric distortion: random brightness, contrast, saturation, hue
# 7. Pack to CHW format for model input
#
# Expected output per sample:
# - img: float32, shape (3, 224, 224)
# - gt_semantic_seg: int32, shape (224, 224) with values 0-149 (and 255 for ignore)
#
# Data structure:
# ===============
# data_root/
# ├── images/
# │   ├── training/          (*.jpg)
# │   └── validation/         (*.jpg)
# └── annotations/
#     ├── training/           (*.png)
#     └── validation/          (*.png)
