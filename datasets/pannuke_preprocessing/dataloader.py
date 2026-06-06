"""DataLoader utilities for PanNuke semantic dataset.

Provides a simple batch iterator compatible with the project's training
pipeline (mirrors ADE20KDataLoader behavior).
"""
from typing import List, Dict
import numpy as np
from .pannuke_dataset import PanNukeDataset
from .preprocessing import build_pipeline
import torch
from PIL import Image


class PanNukeDataLoader:
    def __init__(
        self,
        dataset: PanNukeDataset,
        pipeline,
        batch_size: int = 4,
        shuffle: bool = False,
        drop_last: bool = False,
        worker_init_fn=None,
        generator=None,
    ):
        self.dataset = dataset
        self.pipeline = pipeline
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.worker_init_fn = worker_init_fn
        self.generator = generator

        self.indices = list(range(len(dataset)))
        if shuffle:
            np.random.shuffle(self.indices)

        self.num_batches = len(self.indices) // batch_size
        if not drop_last and len(self.indices) % batch_size != 0:
            self.num_batches += 1

    def __len__(self) -> int:
        return self.num_batches

    @staticmethod
    def _pad_batch(batch_imgs, batch_segs):
        max_h = 0
        max_w = 0
        layouts = []
        for img in batch_imgs:
            if img.ndim == 3 and img.shape[0] <= 4 and img.shape[1] > 4:
                h, w = img.shape[1], img.shape[2]
                is_chw = True
            else:
                h, w = img.shape[0], img.shape[1]
                is_chw = False
            max_h = max(max_h, h)
            max_w = max(max_w, w)
            layouts.append(is_chw)

        padded_imgs = []
        padded_segs = []

        for img, seg, is_chw in zip(batch_imgs, batch_segs, layouts):
            if is_chw:
                c = img.shape[0]
                padded_img = np.zeros((c, max_h, max_w), dtype=img.dtype)
                padded_img[:, :img.shape[1], :img.shape[2]] = img
            else:
                padded_img = np.zeros((max_h, max_w, img.shape[2]), dtype=img.dtype)
                padded_img[:img.shape[0], :img.shape[1]] = img

            padded_seg = np.full((max_h, max_w), 255, dtype=seg.dtype)
            padded_seg[:seg.shape[0], :seg.shape[1]] = seg

            padded_imgs.append(padded_img)
            padded_segs.append(padded_seg)

        return padded_imgs, padded_segs

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            if self.generator is not None:
                indices = torch.randperm(len(indices), generator=self.generator).tolist()
            else:
                indices = np.random.permutation(indices).tolist()

        for batch_idx in range(self.num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(indices))

            if self.drop_last and end_idx - start_idx < self.batch_size:
                continue

            batch_indices = indices[start_idx:end_idx]

            batch_imgs = []
            batch_segs = []

            for idx in batch_indices:
                sample = self.dataset[idx]
                processed = self.pipeline(sample)
                batch_imgs.append(processed['img'])
                batch_segs.append(processed['gt_semantic_seg'])

            batch_imgs, batch_segs = self._pad_batch(batch_imgs, batch_segs)

            batch_data = {
                'img': torch.from_numpy(np.stack(batch_imgs, axis=0)).float(),
                'gt_semantic_seg': torch.from_numpy(np.stack(batch_segs, axis=0)).long(),
            }

            yield batch_data


def create_train_loader(
    data_root: str,
    pipeline_config: List[Dict],
    batch_size: int = 4,
    shuffle: bool = True,
    split: str = 'training',
    **kwargs
) -> PanNukeDataLoader:
    dataset = PanNukeDataset(data_root, split=split)
    pipeline = build_pipeline(pipeline_config)
    return PanNukeDataLoader(dataset, pipeline, batch_size=batch_size, shuffle=shuffle, drop_last=True, **kwargs)


def create_val_loader(
    data_root: str,
    pipeline_config: List[Dict],
    batch_size: int = 1,
    split: str = 'validation',
    **kwargs
) -> PanNukeDataLoader:
    dataset = PanNukeDataset(data_root, split=split)
    pipeline = build_pipeline(pipeline_config)
    return PanNukeDataLoader(dataset, pipeline, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)

