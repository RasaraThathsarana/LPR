"""Dataset availability checks and offline tiling helpers for Inria."""

from __future__ import annotations

import re
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from .inria_dataset import build_tile_coordinates, extract_tile

INRIA_URLS = [
    'https://project.inria.fr/aerialimagelabeling/files/AerialImageDataset.zip',
    'https://project.inria.fr/aerialimagelabeling/files/NEW2-AerialImageDataset.zip',
]
ARCHIVE_NAME = 'AerialImageDataset.zip'
TILE_SIZE = 224
LARGE_IMAGE_THRESHOLD = 512


class DownloadProgressBar(tqdm):
    def update_to(self, blocks: int = 1, block_size: int = 1, total_size: Optional[int] = None):
        if total_size is not None:
            self.total = total_size
        self.update(blocks * block_size - self.n)


def _has_inria_dataset(data_root: Path) -> bool:
    required_paths = [
        data_root / 'training' / 'images',
        data_root / 'training' / 'gt',
        data_root / 'validation' / 'images',
        data_root / 'validation' / 'gt',
    ]
    return all(path.exists() for path in required_paths)


def _download_file(url: str, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)

    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=f'Downloading {target_path.name}') as progress:
        def report_hook(blocks, block_size, total_size=None):
            progress.update_to(blocks, block_size, total_size)

        urllib.request.urlretrieve(url, filename=str(target_path), reporthook=report_hook)


def _extract_zip(zip_path: Path, extract_to: Path) -> None:
    if not zip_path.exists():
        raise FileNotFoundError(f'Archive not found: {zip_path}')

    with zipfile.ZipFile(str(zip_path), 'r') as archive:
        archive.extractall(path=str(extract_to))


def _iter_city_images(directory: Path) -> Iterable[Path]:
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == '.tif'
    )


def _save_tile(image_tile: np.ndarray, mask_tile: np.ndarray, image_path: Path, mask_path: Optional[Path]) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_tile.astype(np.uint8), mode='RGB').save(image_path)
    if mask_path is not None:
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask_tile.astype(np.uint8), mode='L').save(mask_path)


def _prepare_tiled_split(src_images: Path, src_masks: Optional[Path], dst_images: Path, dst_masks: Optional[Path], validation: bool = False) -> None:
    pattern = re.compile(r'([A-Za-z]+)(\d+)')

    for image_path in _iter_city_images(src_images):
        match = pattern.match(image_path.stem)
        index = int(match.group(2)) if match else 0
        if validation and index > 5:
            continue
        if not validation and index <= 5:
            continue

        with Image.open(image_path) as image:
            image_array = np.array(image.convert('RGB'), dtype=np.uint8)

        mask_array = None
        if src_masks is not None:
            mask_path = src_masks / image_path.name
            if mask_path.exists():
                with Image.open(mask_path) as mask:
                    mask_array = np.array(mask.convert('L'), dtype=np.uint8)
                    mask_array = (mask_array > 0).astype(np.uint8)
            else:
                raise FileNotFoundError(f'Expected mask missing for {image_path.name}')

        tile_coords = build_tile_coordinates(
            height=image_array.shape[0],
            width=image_array.shape[1],
            tile_size=TILE_SIZE,
            threshold=LARGE_IMAGE_THRESHOLD,
        )

        for tile_idx, (top, left) in enumerate(tile_coords):
            image_tile = extract_tile(
                image_array,
                top,
                left,
                tile_size=TILE_SIZE,
                pad_value=0,
                image=True,
            )
            mask_tile = None
            if mask_array is not None:
                mask_tile = extract_tile(
                    mask_array,
                    top,
                    left,
                    tile_size=TILE_SIZE,
                    pad_value=255,
                    image=False,
                )

            suffix = f'{image_path.stem}_tile_{tile_idx:04d}.png'
            image_dst = dst_images / suffix
            mask_dst = dst_masks / suffix if dst_masks is not None else None
            _save_tile(image_tile, mask_tile if mask_tile is not None else np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8), image_dst, mask_dst)


def _materialize_expected_layout(raw_root: Path, data_root: Path) -> None:
    """Convert the raw Inria archive layout into tiled train/val/test folders."""
    train_root = raw_root / 'train'
    test_root = raw_root / 'test'

    if not train_root.exists():
        raise RuntimeError(f'Expected train directory not found in {raw_root}')

    _prepare_tiled_split(
        src_images=train_root / 'images',
        src_masks=train_root / 'gt',
        dst_images=data_root / 'training' / 'images',
        dst_masks=data_root / 'training' / 'gt',
        validation=False,
    )

    _prepare_tiled_split(
        src_images=train_root / 'images',
        src_masks=train_root / 'gt',
        dst_images=data_root / 'validation' / 'images',
        dst_masks=data_root / 'validation' / 'gt',
        validation=True,
    )

    if test_root.exists():
        for image_path in _iter_city_images(test_root / 'images'):
            with Image.open(image_path) as image:
                image_array = np.array(image.convert('RGB'), dtype=np.uint8)

            tile_coords = build_tile_coordinates(
                height=image_array.shape[0],
                width=image_array.shape[1],
                tile_size=TILE_SIZE,
                threshold=LARGE_IMAGE_THRESHOLD,
            )
            for tile_idx, (top, left) in enumerate(tile_coords):
                image_tile = extract_tile(
                    image_array,
                    top,
                    left,
                    tile_size=TILE_SIZE,
                    pad_value=0,
                    image=True,
                )
                image_dst = data_root / 'test' / 'images' / f'{image_path.stem}_tile_{tile_idx:04d}.png'
                _save_tile(image_tile, np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8), image_dst, None)


def ensure_inria_dataset(data_root: str, download: bool = False) -> None:
    data_root_path = Path(data_root)
    if _has_inria_dataset(data_root_path):
        return

    if not download:
        raise FileNotFoundError(
            f'Inria dataset not found at {data_root_path}. '
            'Pass --download-data to fetch and prepare it automatically, or '
            'manually place the extracted archive in the expected layout.'
        )

    archive_dir = data_root_path.parent
    archive_path = archive_dir / ARCHIVE_NAME
    staging_dir = archive_dir / '_inria_extract'
    raw_extract_dir = staging_dir / 'AerialImageDataset'

    for url in INRIA_URLS:
        try:
            if not archive_path.exists():
                print(f'Downloading Inria dataset from {url} to {archive_path}...')
                _download_file(url, archive_path)
            break
        except Exception:
            if archive_path.exists():
                archive_path.unlink(missing_ok=True)
            continue
    else:
        raise RuntimeError(
            'Failed to download the Inria archive from the known public URLs. '
            'You may need to download it manually from the dataset homepage.'
        )

    if staging_dir.exists():
        shutil.rmtree(str(staging_dir))

    print(f'Extracting {archive_path} to {staging_dir}...')
    _extract_zip(archive_path, staging_dir)

    if not raw_extract_dir.exists():
        raise RuntimeError(f'Expected extracted dataset at {raw_extract_dir}, but it was not found.')

    if data_root_path.exists():
        shutil.rmtree(str(data_root_path))

    _materialize_expected_layout(raw_extract_dir, data_root_path)

    shutil.rmtree(str(staging_dir), ignore_errors=True)

    if not _has_inria_dataset(data_root_path):
        raise RuntimeError(
            f'Failed to prepare Inria dataset at {data_root_path}. '
            'Please verify the extracted archive contains the expected raw layout.'
        )

    print(f'Inria dataset is ready at {data_root_path}')
