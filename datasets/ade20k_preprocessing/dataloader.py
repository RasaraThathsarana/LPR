"""
DataLoader utilities for batch processing.

This module provides utilities for creating data loaders similar to PyTorch
dataloaders for efficient batch processing of ADE20K data.
"""

from typing import List, Dict
import numpy as np
from .ade20k_dataset import ADE20KDataset
from .preprocessing import build_pipeline
import torch


class ADE20KDataLoader:
    """Simple DataLoader for ADE20K dataset.
    
    Args:
        dataset (ADE20KDataset): Dataset instance
        pipeline: Preprocessing pipeline (Compose object)
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle data
        drop_last (bool): Whether to drop last incomplete batch
    """
    
    def __init__(
        self,
        dataset: ADE20KDataset,
        pipeline,
        batch_size: int = 4,
        shuffle: bool = False,
        drop_last: bool = False
    ):
        self.dataset = dataset
        self.pipeline = pipeline
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Create indices
        self.indices = list(range(len(dataset)))
        if shuffle:
            np.random.shuffle(self.indices)
        
        # Calculate number of batches
        self.num_batches = len(self.indices) // batch_size
        if not drop_last and len(self.indices) % batch_size != 0:
            self.num_batches += 1
    
    def __len__(self) -> int:
        """Number of batches."""
        return self.num_batches

    @staticmethod
    def _pad_batch(batch_imgs, batch_segs):
        """Pad samples in a batch to the same spatial size.

        ADE20K validation uses resize-with-keep-ratio, so image sizes can vary
        across samples. Padding keeps batching simple while preserving the
        original content and masking out padded labels with 255.
        """
        def spatial_shape(img):
            # PackSegInputs yields CHW, but keep a fallback for HWC arrays.
            if img.ndim == 3 and img.shape[0] <= 4 and img.shape[1] > 4:
                return img.shape[1], img.shape[2], True
            return img.shape[0], img.shape[1], False

        max_h = 0
        max_w = 0
        layouts = []
        for img in batch_imgs:
            h, w, is_chw = spatial_shape(img)
            max_h = max(max_h, h)
            max_w = max(max_w, w)
            layouts.append(is_chw)

        padded_imgs = []
        padded_segs = []

        for img, seg, is_chw in zip(batch_imgs, batch_segs, layouts):
            h, w = seg.shape[:2]

            if is_chw:
                c = img.shape[0]
                padded_img = np.zeros((c, max_h, max_w), dtype=img.dtype)
                padded_img[:, :img.shape[1], :img.shape[2]] = img
            else:
                padded_img = np.zeros((max_h, max_w, img.shape[2]), dtype=img.dtype)
                padded_img[:img.shape[0], :img.shape[1]] = img

            padded_seg = np.full((max_h, max_w), 255, dtype=seg.dtype)
            padded_seg[:h, :w] = seg

            padded_imgs.append(padded_img)
            padded_segs.append(padded_seg)

        return padded_imgs, padded_segs
    
    def __iter__(self):
        """Iterate over batches."""
        for batch_idx in range(self.num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(self.indices))
            
            # Skip last batch if drop_last=True and incomplete
            if self.drop_last and end_idx - start_idx < self.batch_size:
                continue
            
            # Get batch indices
            batch_indices = self.indices[start_idx:end_idx]
            
            # Load and preprocess samples
            batch_imgs = []
            batch_segs = []
            
            for idx in batch_indices:
                sample = self.dataset[idx]
                processed = self.pipeline(sample)
                
                batch_imgs.append(processed['img'])
                batch_segs.append(processed['gt_semantic_seg'])

            batch_imgs, batch_segs = self._pad_batch(batch_imgs, batch_segs)
            
            # Stack to create batch
            batch_data = {
                'img': torch.from_numpy(np.stack(batch_imgs, axis=0)).float(),  # (B, C, H, W)
                'gt_semantic_seg': torch.from_numpy(np.stack(batch_segs, axis=0)).long(),  # (B, H, W)
            }
            
            yield batch_data


def create_train_loader(
    data_root: str,
    pipeline_config: List[Dict],
    batch_size: int = 4,
    shuffle: bool = True,
    **kwargs
) -> ADE20KDataLoader:
    """Create training dataloader.
    
    Args:
        data_root (str): Path to dataset root
        pipeline_config (list): Pipeline configuration
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle data
        
    Returns:
        ADE20KDataLoader: Training dataloader
    """
    dataset = ADE20KDataset(data_root, split='training', reduce_zero_label=True)
    pipeline = build_pipeline(pipeline_config)
    
    return ADE20KDataLoader(
        dataset,
        pipeline,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,  # Drop incomplete batches for training
        **kwargs
    )


def create_val_loader(
    data_root: str,
    pipeline_config: List[Dict],
    batch_size: int = 1,
    **kwargs
) -> ADE20KDataLoader:
    """Create validation dataloader.
    
    Args:
        data_root (str): Path to dataset root
        pipeline_config (list): Pipeline configuration
        batch_size (int): Batch size
        
    Returns:
        ADE20KDataLoader: Validation dataloader
    """
    dataset = ADE20KDataset(data_root, split='validation', reduce_zero_label=True)
    pipeline = build_pipeline(pipeline_config)
    
    return ADE20KDataLoader(
        dataset,
        pipeline,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation data
        drop_last=False,  # Keep all validation samples
        **kwargs
    )


# Example usage
if __name__ == '__main__':
    from preprocessing_config import TRAIN_PIPELINE, VAL_PIPELINE, DATA_ROOT
    
    print("Creating training dataloader...")
    train_loader = create_train_loader(DATA_ROOT, TRAIN_PIPELINE, batch_size=4)
    
    print("Creating validation dataloader...")
    val_loader = create_val_loader(DATA_ROOT, VAL_PIPELINE, batch_size=1)
    
    print(f"\nTraining batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Example iteration
    print("\n" + "=" * 60)
    print("Training Batch Example")
    print("=" * 60)
    
    for batch_data in train_loader:
        print(f"Batch images shape: {batch_data['img'].shape}")
        print(f"Batch masks shape: {batch_data['gt_semantic_seg'].shape}")
        break  # Only show first batch
    
    print("\n" + "=" * 60)
    print("Validation Batch Example")
    print("=" * 60)
    
    for batch_data in val_loader:
        print(f"Batch images shape: {batch_data['img'].shape}")
        print(f"Batch masks shape: {batch_data['gt_semantic_seg'].shape}")
        break  # Only show first batch
