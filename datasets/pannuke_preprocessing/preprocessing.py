"""Preprocessing helpers for converting PanNuke instance annotations to
semantic segmentation maps.

The functions are intentionally flexible to handle several common PanNuke
export formats. They attempt to map per-instance labels to per-pixel
semantic class ids using auxiliary JSON/CSV mapping files when present.
"""
from pathlib import Path
from typing import Dict, Optional
import numpy as np
from PIL import Image
import json
import warnings

from .preprocessing_config import DEFAULT_CLASS_MAPPING, INSTANCE_MASK_SUFFIXES

# Reuse the preprocessing transforms builder from the Inria implementation
from ..inria_preprocessing.preprocessing import build_pipeline as _inria_build_pipeline


def build_pipeline(pipeline_config):
    return _inria_build_pipeline(pipeline_config)


def instances_to_semantic(instance_mask: np.ndarray, instance_class_map: Dict[int, int], num_classes: int = 6) -> np.ndarray:
    """Convert an instance mask (H,W) where each pixel's value is an instance id
    into a semantic map (H,W) where pixel values are class ids (0..num_classes-1).

    - `instance_mask` : np.ndarray of integer instance ids (0 for background)
    - `instance_class_map` : mapping {instance_id: class_id}
    - `num_classes` : total classes including background (default 6: 0..5)
    """
    h, w = instance_mask.shape[:2]
    semantic = np.zeros((h, w), dtype=np.uint8)

    # Ensure background stays 0
    unique_ids = np.unique(instance_mask)
    for inst_id in unique_ids:
        if inst_id == 0:
            continue
        cls = instance_class_map.get(int(inst_id), None)
        if cls is None:
            # default to class 1 if unknown (neoplastic)
            cls = 1
        if not (0 <= cls < num_classes):
            warnings.warn(f"Class id {cls} outside range, clipping")
            cls = max(0, min(num_classes - 1, int(cls)))
        mask = instance_mask == inst_id
        semantic[mask] = int(cls)

    return semantic


def _load_instance_mask(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == '.npy':
        return np.load(path)
    if suffix == '.npz':
        arr = np.load(path)
        # prefer first array inside
        for v in arr.values():
            return v
        raise ValueError('Empty npz')
    # Fallback to image formats; assume integer ids encoded in PNG
    img = Image.open(path)
    arr = np.array(img)
    # If RGB, convert to single id by taking first channel
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def _load_instance_class_map(path: Path) -> Optional[Dict[int, int]]:
    if not path.exists():
        return None
    try:
        if path.suffix.lower() in ('.json',):
            with open(path, 'r') as f:
                data = json.load(f)
            # Support either dict of {inst_id: class_id} or list of objects
            if isinstance(data, dict):
                return {int(k): int(v) for k, v in data.items()}
            if isinstance(data, list):
                mapping = {}
                for obj in data:
                    # obj could be {"id":1, "class":2}
                    if 'id' in obj and ('class' in obj or 'label' in obj):
                        mapping[int(obj['id'])] = int(obj.get('class', obj.get('label')))
                return mapping
    except Exception:
        warnings.warn(f"Failed to parse class mapping: {path}")
    return None


def create_semantic_masks_from_instances(
    raw_ann_dir: str,
    out_dir: str,
    instance_suffixes: tuple = INSTANCE_MASK_SUFFIXES,
    num_classes: int = 6,
    default_class_mapping: Dict[int, int] = DEFAULT_CLASS_MAPPING,
):
    """Scan `raw_ann_dir` for instance annotation files and create semantic
    segmentation PNGs in `out_dir`.

    For each instance mask file named `<name>.<suf>` the function will look
    for a class mapping file named `<name>_classes.json` (optional). If it's
    missing the `default_class_mapping` will be used for mapping instance
    labels to semantic class ids.
    """
    raw_ann_dir = Path(raw_ann_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = []
    for suf in instance_suffixes:
        files.extend(list(raw_ann_dir.glob(f"*{suf}")))

    if not files:
        raise FileNotFoundError(f"No instance mask files found in {raw_ann_dir}")

    for f in sorted(files):
        name = f.stem
        inst_mask = _load_instance_mask(f)

        # try to load mapping file
        mapping_path = f.with_name(name + '_classes.json')
        mapping = _load_instance_class_map(mapping_path) or default_class_mapping

        semantic = instances_to_semantic(inst_mask, mapping, num_classes=num_classes)

        out_path = out_dir / (name + '.png')
        # Save as uint8 PNG
        Image.fromarray(semantic.astype('uint8')).save(out_path)

    return True
