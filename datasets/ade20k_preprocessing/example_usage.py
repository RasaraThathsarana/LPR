"""
Example usage of ADE20K dataset with preprocessing.

This example shows how to:
1. Load the dataset
2. Apply preprocessing pipeline
3. Prepare data like in MMSegmentation
"""

import numpy as np
from ade20k_dataset import ADE20KDataset
from preprocessing import build_pipeline
from preprocessing_config import TRAIN_PIPELINE, VAL_PIPELINE, DATA_ROOT


def example_training_data():
    """Example of loading training data with preprocessing."""
    print("=" * 60)
    print("Example: Training Data with Preprocessing")
    print("=" * 60)
    
    # Initialize dataset
    dataset = ADE20KDataset(
        data_root=DATA_ROOT,
        split='training',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        reduce_zero_label=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of classes: {dataset.get_num_classes()}")
    print(f"Class names: {dataset.get_classes()[:5]}... (showing first 5)")
    
    # Build preprocessing pipeline
    pipeline = build_pipeline(TRAIN_PIPELINE)
    
    # Get a sample
    sample = dataset[0]
    print(f"\nOriginal sample:")
    print(f"  Image shape: {sample['img'].shape}")
    print(f"  Image dtype: {sample['img'].dtype}")
    print(f"  Seg map shape: {sample['gt_semantic_seg'].shape}")
    print(f"  Seg map dtype: {sample['gt_semantic_seg'].dtype}")
    print(f"  Seg map range: [{sample['gt_semantic_seg'].min()}, {sample['gt_semantic_seg'].max()}]")
    
    # Apply preprocessing
    processed_sample = pipeline(sample)
    print(f"\nAfter preprocessing:")
    print(f"  Image shape: {processed_sample['img'].shape}")  # Should be (3, 512, 512)
    print(f"  Image dtype: {processed_sample['img'].dtype}")
    print(f"  Seg map shape: {processed_sample['gt_semantic_seg'].shape}")  # Should be (512, 512)
    print(f"  Seg map dtype: {processed_sample['gt_semantic_seg'].dtype}")
    print(f"  Seg map range: [{processed_sample['gt_semantic_seg'].min()}, {processed_sample['gt_semantic_seg'].max()}]")
    

def example_validation_data():
    """Example of loading validation data with preprocessing."""
    print("\n" + "=" * 60)
    print("Example: Validation Data with Preprocessing")
    print("=" * 60)
    
    # Initialize dataset
    dataset = ADE20KDataset(
        data_root=DATA_ROOT,
        split='validation',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        reduce_zero_label=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Build preprocessing pipeline (no augmentation for validation)
    pipeline = build_pipeline(VAL_PIPELINE)
    
    # Get a sample
    sample = dataset[0]
    print(f"\nOriginal sample:")
    print(f"  Image shape: {sample['img'].shape}")
    
    # Apply preprocessing
    processed_sample = pipeline(sample)
    print(f"\nAfter preprocessing (no augmentation):")
    print(f"  Image shape: {processed_sample['img'].shape}")
    print(f"  Seg map shape: {processed_sample['gt_semantic_seg'].shape}")


def example_batch_loading():
    """Example of creating batches of data."""
    print("\n" + "=" * 60)
    print("Example: Batch Loading")
    print("=" * 60)
    
    dataset = ADE20KDataset(
        data_root=DATA_ROOT,
        split='training',
        reduce_zero_label=True
    )
    
    pipeline = build_pipeline(TRAIN_PIPELINE)
    
    batch_size = 4
    batch_imgs = []
    batch_segs = []
    
    for i in range(min(batch_size, len(dataset))):
        sample = dataset[i]
        processed = pipeline(sample)
        batch_imgs.append(processed['img'])
        batch_segs.append(processed['gt_semantic_seg'])
    
    # Stack to create batch
    # Note: All images in batch should have same size for stacking
    batch_imgs = np.stack(batch_imgs, axis=0)  # (B, C, H, W)
    batch_segs = np.stack(batch_segs, axis=0)  # (B, H, W)
    
    print(f"Batch shape:")
    print(f"  Images: {batch_imgs.shape}")  # Should be (4, 3, 512, 512)
    print(f"  Seg maps: {batch_segs.shape}")  # Should be (4, 512, 512)
    print(f"  Images dtype: {batch_imgs.dtype}")
    print(f"  Seg maps dtype: {batch_segs.dtype}")


def example_data_consistency():
    """Example showing data is same as in MMSegmentation."""
    print("\n" + "=" * 60)
    print("Data Consistency Check")
    print("=" * 60)
    
    dataset = ADE20KDataset(DATA_ROOT, split='training', reduce_zero_label=True)
    pipeline = build_pipeline(TRAIN_PIPELINE)
    
    # Check multiple samples
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        processed = pipeline(sample)
        
        print(f"\nSample {i}:")
        print(f"  Image valid range: [{processed['img'].mean():.2f} ± {processed['img'].std():.2f}]")
        print(f"  Seg map classes: {np.unique(processed['gt_semantic_seg'])[:5]}...")
        print(f"  Output shapes match expected: {processed['img'].shape == (3, 512, 512)} and {processed['gt_semantic_seg'].shape == (512, 512)}")


if __name__ == '__main__':
    print("\n★ ADE20K Dataset Preprocessing Examples ★\n")
    
    try:
        example_training_data()
        example_validation_data()
        example_batch_loading()
        example_data_consistency()
        
        print("\n" + "=" * 60)
        print("✓ All examples completed successfully!")
        print("=" * 60)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure the dataset path is correct:")
        print(f"Expected: {DATA_ROOT}")
        print("\nDataset structure should be:")
        print("data_root/")
        print("├── images/")
        print("│   ├── training/          (*.jpg)")
        print("│   └── validation/         (*.jpg)")
        print("└── annotations/")
        print("    ├── training/           (*.png)")
        print("    └── validation/          (*.png)")
