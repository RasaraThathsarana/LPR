"""
ADE20K Preprocessing Pipeline Configuration.

This configuration produces the exact same preprocessing as MMSegmentation
when using the configs/_base_/datasets/ade20k.py configuration.
"""

# Dataset settings
DATASET_TYPE = 'ADE20KDataset'
DATA_ROOT = 'data/ade/ADEChallengeData2016'
CROP_SIZE = (512, 512)

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

# Validation/Test pipeline
# Note: No augmentation, only resizing and packing
VAL_PIPELINE = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
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
# 4. Random crop: crop to 512x512 with max 75% of single class
# 5. Random horizontal flip: 50% probability
# 6. Photometric distortion: random brightness, contrast, saturation, hue
# 7. Pack to CHW format for model input
#
# Expected output per sample:
# - img: float32, shape (3, 512, 512)
# - gt_semantic_seg: int32, shape (512, 512) with values 0-149 (and 255 for ignore)
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
