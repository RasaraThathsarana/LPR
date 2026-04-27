"""Legacy helpers for pre-tiled Inria-style directories.

The main training path now lives under ``datasets.inria_preprocessing``.
This module remains as a small compatibility layer for callers that already
have pre-split 224x224 tile folders on disk.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class InriaTileDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.image_names = sorted(
            f.name for f in self.images_dir.iterdir()
            if f.is_file() and f.suffix.lower() in {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}
        )

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = self.images_dir / img_name
        mask_path = self.masks_dir / img_name

        image = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.uint8)
        mask = (mask > 0).astype(np.int64)

        if self.transform is not None:
            image, mask = self.transform(image=image, mask=mask)

        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        mask = torch.from_numpy(mask).long().unsqueeze(0)
        return image, mask


def get_224_dataloaders(train_img_dir, train_mask_dir, batch_size=32, num_workers: int = 4):
    train_dataset = InriaTileDataset(images_dir=train_img_dir, masks_dir=train_mask_dir)
    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

