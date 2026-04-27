"""Dataset availability checks and offline tiling helpers for Inria."""

from __future__ import annotations

import re
import shutil
import subprocess
import urllib.request
import urllib.error
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
    'https://files.inria.fr/aerialimagelabeling/getAerial.sh'
]
ARCHIVE_NAME = 'AerialImageDataset.zip'
TILE_SIZE = 224
LARGE_IMAGE_THRESHOLD = 512
MIN_FREE_BYTES = 45 * 1024**3


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


def _count_tiles(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for path in directory.iterdir() if path.is_file() and path.suffix.lower() == '.png')


def _validate_prepared_layout(data_root: Path) -> None:
    required_dirs = [
        data_root / 'training' / 'images',
        data_root / 'training' / 'gt',
        data_root / 'validation' / 'images',
        data_root / 'validation' / 'gt',
    ]
    missing = [str(path) for path in required_dirs if not path.exists()]
    if missing:
        raise RuntimeError(
            'Prepared Inria dataset is missing required directories: '
            + ', '.join(missing)
        )

    train_images = _count_tiles(data_root / 'training' / 'images')
    train_masks = _count_tiles(data_root / 'training' / 'gt')
    val_images = _count_tiles(data_root / 'validation' / 'images')
    val_masks = _count_tiles(data_root / 'validation' / 'gt')

    if train_images == 0 or train_masks == 0 or val_images == 0 or val_masks == 0:
        raise RuntimeError(
            'Prepared Inria dataset is empty or incomplete after tiling. '
            f'train(images={train_images}, masks={train_masks}), '
            f'val(images={val_images}, masks={val_masks}).'
        )


def _find_raw_inria_root(search_root: Path) -> Optional[Path]:
    """Find the raw Inria folder that contains train/test splits."""
    candidates = [search_root]
    try:
        candidates.extend(
            sorted(
                path for path in search_root.rglob('*')
                if path.is_dir()
            )
        )
    except OSError:
        pass

    for candidate in candidates:
        train_root = candidate / 'train'
        test_root = candidate / 'test'
        if train_root.exists() and test_root.exists():
            if (train_root / 'images').exists() and ((train_root / 'gt').exists() or (candidate / 'gt').exists()):
                return candidate
            if (candidate / 'training' / 'images').exists():
                return candidate
    return None


def _download_file(url: str, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)

    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=f'Downloading {target_path.name}') as progress:
        request = urllib.request.Request(
            url,
            headers={
                'User-Agent': (
                    'Mozilla/5.0 (X11; Linux x86_64) '
                    'AppleWebKit/537.36 (KHTML, like Gecko) '
                    'Chrome/125.0 Safari/537.36'
                ),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Referer': 'https://project.inria.fr/aerialimagelabeling/',
            },
        )

        try:
            with urllib.request.urlopen(request) as response, open(target_path, 'wb') as out_file:
                total_size = response.headers.get('Content-Length')
                if total_size is not None:
                    progress.total = int(total_size)

                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    progress.update(len(chunk))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f'HTTP error {exc.code} while downloading {url}') from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f'Network error while downloading {url}: {exc.reason}') from exc


def _extract_zip(zip_path: Path, extract_to: Path) -> None:
    if not zip_path.exists():
        raise FileNotFoundError(f'Archive not found: {zip_path}')

    with zipfile.ZipFile(str(zip_path), 'r') as archive:
        archive.extractall(path=str(extract_to))


def _has_7z() -> bool:
    return shutil.which('7z') is not None or shutil.which('7za') is not None


def _ensure_free_space(target_dir: Path, required_bytes: int = MIN_FREE_BYTES) -> None:
    usage = shutil.disk_usage(str(target_dir))
    if usage.free < required_bytes:
        raise RuntimeError(
            f'Not enough free disk space at {target_dir}: '
            f'need at least {required_bytes / 1024**3:.1f} GiB free, '
            f'found {usage.free / 1024**3:.1f} GiB. '
            'The official Inria archive is very large, so Kaggle-style environments '
            'usually need a mounted dataset or a pre-downloaded archive in '
            '`--inria-archive`.'
        )


def _looks_like_shell_script(path: Path) -> bool:
    try:
        with path.open('rb') as handle:
            head = handle.read(2048)
    except OSError:
        return False
    return head.startswith(b'#!') or b'getAerial' in head or b'7z' in head or b'curl' in head


def _run_shell_script(script_path: Path, workdir: Path) -> None:
    if not _has_7z():
        raise RuntimeError(
            'The Inria download endpoint returned a shell script that requires `7z` '
            'to extract the split archive, but `7z` is not installed in this environment. '
            'Install `p7zip-full` or provide `--inria-archive` with an already extracted '
            'AerialImageDataset folder or archive.'
        )

    script_path.chmod(script_path.stat().st_mode | 0o111)
    subprocess.run(['bash', str(script_path.name)], cwd=str(workdir), check=True)


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
    raw_root = _find_raw_inria_root(raw_root) or raw_root
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

    _validate_prepared_layout(data_root)


def ensure_inria_dataset(data_root: str, download: bool = False) -> None:
    return ensure_inria_dataset_from_source(data_root, download=download, archive_path=None)


def ensure_inria_dataset_from_source(
    data_root: str,
    download: bool = False,
    archive_path: Optional[str] = None,
) -> None:
    data_root_path = Path(data_root)
    if _has_inria_dataset(data_root_path):
        return

    _ensure_free_space(data_root_path.parent)

    if archive_path:
        archive_input = Path(archive_path)
        if archive_input.is_dir():
            if data_root_path.exists():
                shutil.rmtree(str(data_root_path))
            _materialize_expected_layout(archive_input, data_root_path)
            print(f'Inria dataset is ready at {data_root_path}')
            return

        if not archive_input.exists():
            raise FileNotFoundError(f'Inria archive not found: {archive_input}')

        archive_dir = data_root_path.parent
        staging_dir = archive_dir / '_inria_extract'
        raw_extract_dir = staging_dir / 'AerialImageDataset'

        if staging_dir.exists():
            shutil.rmtree(str(staging_dir))
        staging_dir.mkdir(parents=True, exist_ok=True)

        try:
            if zipfile.is_zipfile(str(archive_input)):
                _extract_zip(archive_input, staging_dir)
            elif _looks_like_shell_script(archive_input):
                if data_root_path.exists():
                    shutil.rmtree(str(data_root_path))
                script_copy = archive_dir / 'getAerial.sh'
                shutil.copy2(archive_input, script_copy)
                _run_shell_script(script_copy, archive_dir)
                script_copy.unlink(missing_ok=True)
            else:
                raise RuntimeError(
                    f'Unsupported Inria archive format: {archive_input}. '
                    'Expected a zip archive, a getAerial.sh script, or an extracted AerialImageDataset folder.'
                )

            if not raw_extract_dir.exists():
                detected = _find_raw_inria_root(staging_dir) or _find_raw_inria_root(archive_dir)
                if detected is None:
                    raise RuntimeError(
                        f'Expected extracted dataset at {raw_extract_dir}, but it was not found and no raw Inria root could be detected.'
                    )
                raw_extract_dir = detected
            if data_root_path.exists():
                shutil.rmtree(str(data_root_path))
            _materialize_expected_layout(raw_extract_dir, data_root_path)
        finally:
            shutil.rmtree(str(staging_dir), ignore_errors=True)

        if _has_inria_dataset(data_root_path):
            _validate_prepared_layout(data_root_path)
            print(f'Inria dataset is ready at {data_root_path}')
            return
        raise RuntimeError(
            f'Failed to prepare Inria dataset from archive {archive_input}. '
            'Please verify the archive contains the expected raw layout.'
        )

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
                if archive_path.stat().st_size == 0:
                    raise RuntimeError(f'Downloaded archive is empty: {archive_path}')
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

    if zipfile.is_zipfile(str(archive_path)):
        print(f'Extracting {archive_path} to {staging_dir}...')
        _extract_zip(archive_path, staging_dir)
    elif _looks_like_shell_script(archive_path):
        script_copy = archive_dir / 'getAerial.sh'
        shutil.copy2(archive_path, script_copy)
        print(f'Running Inria download script: {script_copy}')
        _run_shell_script(script_copy, archive_dir)
        script_copy.unlink(missing_ok=True)
    else:
        raise RuntimeError(
            f'Downloaded file {archive_path} is not a zip archive or recognized shell script. '
            'The Inria site may have changed its download flow.'
        )

    detected_raw_root = _find_raw_inria_root(staging_dir) or _find_raw_inria_root(archive_dir)
    if detected_raw_root is None:
        raise RuntimeError(
            f'Expected extracted dataset at {raw_extract_dir}, but no raw Inria root could be detected in {staging_dir} or {archive_dir}.'
        )

    if data_root_path.exists():
        shutil.rmtree(str(data_root_path))

    _materialize_expected_layout(detected_raw_root, data_root_path)

    shutil.rmtree(str(staging_dir), ignore_errors=True)

    _validate_prepared_layout(data_root_path)

    print(f'Inria dataset is ready at {data_root_path}')
