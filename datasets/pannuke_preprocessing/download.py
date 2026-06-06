"""Helpers to download or build PanNuke data.

This module supports both manual preparation and optional Hugging Face
`datasets` downloading from the `RationAI/PanNuke` mirror.
"""
from pathlib import Path
from typing import Optional
import importlib
import sys
import numpy as np
from PIL import Image

DATASET_NAME = 'RationAI/PanNuke'
HF_SPLITS = ('fold1', 'fold2', 'fold3')


def _import_hf_datasets():
    """Import Hugging Face `datasets` while avoiding the local repo `datasets` package."""
    repo_datasets_dir = Path(__file__).resolve().parents[1]
    cwd = Path.cwd().resolve()
    local_dataset_keys = [k for k in sys.modules if k == 'datasets' or k.startswith('datasets.')]
    saved_modules = {k: sys.modules.pop(k) for k in local_dataset_keys}
    original_sys_path = list(sys.path)

    safe_sys_path = []
    for entry in sys.path:
        if entry in ('', '.', None):
            continue
        try:
            resolved = Path(entry).resolve()
        except Exception:
            continue
        if resolved == repo_datasets_dir or resolved == cwd:
            continue
        safe_sys_path.append(str(resolved))
    sys.path[:] = safe_sys_path
    hf = importlib.import_module('datasets')
    return hf, saved_modules, original_sys_path


def _restore_local_datasets(saved_modules, original_sys_path):
    sys.path[:] = original_sys_path
    for k in list(sys.modules):
        if (k == 'datasets' or k.startswith('datasets.')) and k not in saved_modules:
            sys.modules.pop(k, None)
    sys.modules.update(saved_modules)


def download_instructions() -> str:
    return (
        'PanNuke dataset preparation options:\n'
        '1. Manual preparation:\n'
        '   - Place images under `data/pannuke/images/<split>`\n'
        '   - Place semantic masks under `data/pannuke/semantic_annotations/<split>`\n'
        '   - Supported split names are `training`, `validation`, or Hugging Face folds `fold1`, `fold2`, `fold3`.\n'
        '2. Hugging Face download (recommended):\n'
        '   - Install the datasets library: `pip install datasets`\n'
        '   - Call `ensure_pannuke_dataset(root, download=True)` or use `tools/pannuke_prepare.py --download`.\n'
        '3. If download fails, use a local mirror or cache.\n'
    )


def _has_pannuke_dataset(data_root: Path) -> bool:
    images_dir = data_root / 'images'
    sem_dir = data_root / 'semantic_annotations'
    if not (images_dir.exists() and sem_dir.exists()):
        return False

    has_images = any(images_dir.rglob('*.png')) or any(images_dir.rglob('*.jpg')) or any(images_dir.rglob('*.jpeg'))
    has_sems = any(sem_dir.rglob('*.png')) or any(sem_dir.rglob('*.jpg')) or any(sem_dir.rglob('*.jpeg'))
    return has_images and has_sems


def _convert_hf_example(example, image_path: Path, sem_path: Path) -> None:
    image = example['image']
    instances = example['instances']
    categories = example['categories']

    image_array = np.array(image)
    height, width = image_array.shape[:2]

    semantic = np.zeros((height, width), dtype=np.uint8)
    for mask, category in zip(instances, categories):
        mask_array = np.array(mask)
        if mask_array.ndim == 3:
            mask_array = mask_array[..., 0]
        semantic[mask_array > 0] = int(category) + 1

    image.save(str(image_path))
    Image.fromarray(semantic).save(str(sem_path))


def _prepare_hf_pannuke_dataset(data_root: Path) -> None:
    try:
        hf_datasets, saved_modules, original_sys_path = _import_hf_datasets()
        load_dataset = hf_datasets.load_dataset
    except ImportError:
        raise ImportError(
            'The `datasets` library is required to download PanNuke from Hugging Face. '
            'Install it with `pip install datasets`.'
        )

    try:
        for split in HF_SPLITS:
            images_dir = data_root / 'images' / split
            sem_dir = data_root / 'semantic_annotations' / split
            images_dir.mkdir(parents=True, exist_ok=True)
            sem_dir.mkdir(parents=True, exist_ok=True)

            hf_ds = load_dataset(DATASET_NAME, split=split)
            for idx, example in enumerate(hf_ds):
                image_name = f'{split}_{idx:06d}.png'
                image_path = images_dir / image_name
                sem_path = sem_dir / image_name
                if image_path.exists() and sem_path.exists():
                    continue
                _convert_hf_example(example, image_path, sem_path)
    finally:
        _restore_local_datasets(saved_modules, original_sys_path)


def ensure_pannuke_dataset(data_root: str, download: bool = False) -> None:
    data_root_path = Path(data_root)
    if _has_pannuke_dataset(data_root_path):
        return

    if not download:
        raise FileNotFoundError(
            f'PanNuke dataset not found at {data_root}.\n'
            'Use `download=True` to obtain the dataset from Hugging Face if available, '
            'or prepare it manually using local files.'
        )

    _prepare_hf_pannuke_dataset(data_root_path)


def ensure_data_available(data_root: str) -> bool:
    return _has_pannuke_dataset(Path(data_root))
