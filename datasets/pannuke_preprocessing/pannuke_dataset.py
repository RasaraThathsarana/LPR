"""PanNuke Dataset for semantic segmentation.

This dataset expects a prepared folder with images and a parallel
`semantic_annotations` folder containing per-image semantic masks (PNG).
If you haven't generated semantic masks yet, use the helper
`create_semantic_masks_from_instances` in this package to convert
instance annotations to semantic maps.
"""
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image


class PanNukeDataset:
    """PanNuke dataset for semantic segmentation.

    Directory layout expected:
    data_root/
      images/
        training/
        validation/
      semantic_annotations/
        training/
        validation/
    """

    # PanNuke (nuclei) classes: background + 5 tissue categories
    CLASSES = (
        'background',
        'neoplastic',
        'inflammatory',
        'connective',
        'dead',
        'epithelial',
    )

    # Palette (simple distinct colors) for visualization
    PALETTE = [
        [0, 0, 0],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
    ]

    def __init__(
        self,
        data_root: str,
        split: str = 'training',
        img_suffix: str = '.png',
        seg_map_suffix: str = '.png',
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.img_suffix = img_suffix
        self.seg_map_suffix = seg_map_suffix

        self.img_dir = self.data_root / 'images' / split
        self.ann_dir = self.data_root / 'semantic_annotations' / split

        if not self.img_dir.exists() or not self.ann_dir.exists():
            alias_map = {
                'training': 'fold1',
                'validation': 'fold2',
            }
            alias = alias_map.get(split)
            if alias:
                alt_img_dir = self.data_root / 'images' / alias
                alt_ann_dir = self.data_root / 'semantic_annotations' / alias
                if alt_img_dir.exists() and alt_ann_dir.exists():
                    self.split = alias
                    self.img_dir = alt_img_dir
                    self.ann_dir = alt_ann_dir

        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
        if not self.ann_dir.exists():
            raise FileNotFoundError(f"Annotation directory not found: {self.ann_dir}")

        self.img_files = sorted([f for f in self.img_dir.iterdir() if f.suffix == img_suffix])
        if not self.img_files:
            raise ValueError(f"No images found in {self.img_dir}")

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, idx: int) -> Dict:
        img_file = self.img_files[idx]
        name = img_file.stem

        img = Image.open(img_file).convert('RGB')
        img = np.array(img, dtype=np.uint8)

        seg_file = self.ann_dir / (name + self.seg_map_suffix)
        if not seg_file.exists():
            raise FileNotFoundError(f"Semantic map not found: {seg_file}")

        seg = np.array(Image.open(seg_file), dtype=np.int32)

        return {
            'img': img,
            'gt_semantic_seg': seg,
            'img_path': str(img_file),
            'seg_map_path': str(seg_file),
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
