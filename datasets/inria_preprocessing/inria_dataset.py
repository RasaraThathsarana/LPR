"""Inria Aerial Image Labeling dataset helpers.

After the dataset is downloaded, the raw images are tiled offline into
224x224 patches. The dataset class below reads those tiles directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = ('.tif', '.tiff', '.png', '.jpg', '.jpeg')
MASK_SUFFIXES = ('.png', '.tif', '.tiff', '.jpg', '.jpeg')


def _axis_positions(size: int, tile_size: int, stride: int) -> List[int]:
    if size <= tile_size:
        return [0]

    positions = list(range(0, max(size - tile_size + 1, 1), stride))
    last_pos = size - tile_size
    if positions[-1] != last_pos:
        positions.append(last_pos)
    return positions


def build_tile_coordinates(
    height: int,
    width: int,
    tile_size: int = 224,
    threshold: int = 512,
    stride: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """Build top-left tile coordinates for a large image."""
    if max(height, width) <= threshold:
        return [(0, 0)]

    stride = stride or tile_size
    y_positions = _axis_positions(height, tile_size, stride)
    x_positions = _axis_positions(width, tile_size, stride)
    return [(top, left) for top in y_positions for left in x_positions]


def _pad_tile(tile: np.ndarray, tile_size: int, pad_value: int, image: bool) -> np.ndarray:
    pad_h = tile_size - tile.shape[0]
    pad_w = tile_size - tile.shape[1]
    if pad_h <= 0 and pad_w <= 0:
        return tile

    import cv2

    if image:
        return cv2.copyMakeBorder(
            tile,
            0,
            max(pad_h, 0),
            0,
            max(pad_w, 0),
            cv2.BORDER_REFLECT_101,
        )

    return cv2.copyMakeBorder(
        tile,
        0,
        max(pad_h, 0),
        0,
        max(pad_w, 0),
        cv2.BORDER_CONSTANT,
        value=pad_value,
    )


def extract_tile(
    array: np.ndarray,
    top: int,
    left: int,
    tile_size: int = 224,
    pad_value: int = 255,
    image: bool = False,
) -> np.ndarray:
    """Crop a tile from an array and pad it to the requested size if needed."""
    bottom = min(top + tile_size, array.shape[0])
    right = min(left + tile_size, array.shape[1])
    tile = array[top:bottom, left:right]
    return _pad_tile(tile, tile_size, pad_value=pad_value, image=image)


def _find_directory(root: Path, candidates: Sequence[str]) -> Optional[Path]:
    for candidate in candidates:
        path = root / candidate
        if path.exists():
            return path
    return None


def _resolve_mask_path(mask_dir: Path, image_stem: str) -> Optional[Path]:
    for suffix in MASK_SUFFIXES:
        candidate = mask_dir / f'{image_stem}{suffix}'
        if candidate.exists():
            return candidate
    return None


class InriaAerialImageDataset:
    """Dataset wrapper for the pre-tiled Inria Aerial Image Labeling dataset."""

    CLASSES = ('background', 'building')
    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(
        self,
        data_root: str,
        split: str = 'training',
        image_subdir: str = 'images',
        mask_subdir: str = 'gt',
    ):
        self.data_root = Path(data_root)
        self.split = split

        split_dir = _find_directory(self.data_root, [split, split.lower(), split.capitalize()])
        if split_dir is None:
            split_dir = self.data_root / split

        self.img_dir = _find_directory(split_dir, [image_subdir, 'img', 'images'])
        self.ann_dir = _find_directory(split_dir, [mask_subdir, 'gt', 'masks', 'annotations'])

        if self.img_dir is None or not self.img_dir.exists():
            raise FileNotFoundError(f'Image directory not found for Inria dataset: {split_dir / image_subdir}')

        self.image_files = sorted(
            path for path in self.img_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not self.image_files:
            raise ValueError(f'No image files found in {self.img_dir}')

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict:
        img_file = self.image_files[idx]
        img_name = img_file.stem
        mask_file = None

        with Image.open(img_file) as image:
            img = np.array(image.convert('RGB'), dtype=np.uint8)

        if self.ann_dir is not None:
            mask_file = _resolve_mask_path(self.ann_dir, img_name)
            if mask_file is None:
                raise FileNotFoundError(f'Mask not found for image {img_name}')
            with Image.open(mask_file) as mask:
                gt = np.array(mask.convert('L'), dtype=np.uint8)
        else:
            gt = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        gt = (gt > 0).astype(np.int32)

        return {
            'img': img,
            'gt_semantic_seg': gt,
            'img_path': str(img_file),
            'seg_map_path': str(mask_file) if mask_file is not None else None,
        }

    @classmethod
    def get_num_classes(cls) -> int:
        return len(cls.CLASSES)

    @classmethod
    def get_classes(cls) -> Tuple[str, ...]:
        return cls.CLASSES

    @classmethod
    def get_palette(cls) -> List[List[int]]:
        return cls.PALETTE


def stitch_tile_logits(
    tile_logits: Sequence[np.ndarray],
    tile_coords: Sequence[Tuple[int, int]],
    image_shape: Tuple[int, int],
    tile_size: int = 224,
) -> np.ndarray:
    """Merge tile logits back into a full-resolution prediction map."""
    height, width = image_shape
    if not tile_logits:
        return np.zeros((height, width), dtype=np.uint8)

    num_classes = tile_logits[0].shape[0]
    canvas = np.zeros((num_classes, height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)

    for logits, (top, left) in zip(tile_logits, tile_coords):
        tile_h = min(tile_size, height - top)
        tile_w = min(tile_size, width - left)
        canvas[:, top:top + tile_h, left:left + tile_w] += logits[:, :tile_h, :tile_w]
        counts[top:top + tile_h, left:left + tile_w] += 1.0

    counts = np.maximum(counts, 1.0)
    canvas /= counts[None, :, :]
    return canvas.argmax(axis=0).astype(np.uint8)
