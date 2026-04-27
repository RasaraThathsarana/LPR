"""DataLoader utilities for the Inria Aerial Image Labeling dataset."""

from typing import List, Dict

import numpy as np
import torch

from .inria_dataset import InriaAerialImageDataset
from .preprocessing import build_pipeline


class InriaDataLoader:
    def __init__(
        self,
        dataset: InriaAerialImageDataset,
        pipeline,
        batch_size: int = 4,
        shuffle: bool = False,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.pipeline = pipeline
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.indices = list(range(len(dataset)))
        if shuffle:
            np.random.shuffle(self.indices)

        self.num_batches = len(self.indices) // batch_size
        if not drop_last and len(self.indices) % batch_size != 0:
            self.num_batches += 1

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        for batch_idx in range(self.num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(self.indices))

            if self.drop_last and end_idx - start_idx < self.batch_size:
                continue

            batch_indices = self.indices[start_idx:end_idx]
            batch_imgs = []
            batch_segs = []

            for idx in batch_indices:
                sample = self.dataset[idx]
                processed = self.pipeline(sample)
                batch_imgs.append(processed['img'])
                batch_segs.append(processed['gt_semantic_seg'])

            yield {
                'img': torch.from_numpy(np.stack(batch_imgs, axis=0)).float(),
                'gt_semantic_seg': torch.from_numpy(np.stack(batch_segs, axis=0)).long(),
            }


def create_train_loader(
    data_root: str,
    pipeline_config: List[Dict],
    batch_size: int = 4,
    shuffle: bool = True,
    **kwargs,
) -> InriaDataLoader:
    dataset = InriaAerialImageDataset(data_root, split='training')
    pipeline = build_pipeline(pipeline_config)
    return InriaDataLoader(
        dataset,
        pipeline,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        **kwargs,
    )


def create_val_loader(
    data_root: str,
    pipeline_config: List[Dict],
    batch_size: int = 1,
    **kwargs,
) -> InriaDataLoader:
    dataset = InriaAerialImageDataset(data_root, split='validation')
    pipeline = build_pipeline(pipeline_config)
    return InriaDataLoader(
        dataset,
        pipeline,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **kwargs,
    )

