"""Preprocessing transforms for the Inria Aerial Image Labeling dataset."""

from typing import Dict, List, Tuple

import cv2
import numpy as np
import random


class Compose:
    def __init__(self, transforms: List['Transform']):
        self.transforms = transforms

    def __call__(self, data: Dict) -> Dict:
        for transform in self.transforms:
            data = transform(data)
        return data


class Resize:
    def __init__(self, scale: Tuple[int, int], keep_ratio: bool = False):
        self.scale = scale
        self.keep_ratio = keep_ratio

    def __call__(self, data: Dict) -> Dict:
        img = data['img']
        seg_map = data['gt_semantic_seg']

        if self.keep_ratio:
            h, w = img.shape[:2]
            max_long_edge = max(self.scale)
            max_short_edge = min(self.scale)
            scale_factor = min(max_long_edge / max(h, w), max_short_edge / min(h, w))
            new_w = max(1, int(round(w * scale_factor)))
            new_h = max(1, int(round(h * scale_factor)))
        else:
            new_w, new_h = self.scale

        data['img'] = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        data['gt_semantic_seg'] = cv2.resize(
            seg_map.astype(np.float32),
            (new_w, new_h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.int32)
        return data


class RandomCrop:
    def __init__(self, crop_size: Tuple[int, int], cat_max_ratio: float = 1.0):
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio

    def __call__(self, data: Dict) -> Dict:
        img = data['img']
        seg_map = data['gt_semantic_seg']

        h, w = img.shape[:2]
        crop_h, crop_w = self.crop_size

        if h < crop_h or w < crop_w:
            pad_h = max(0, crop_h - h)
            pad_w = max(0, crop_w - w)
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            seg_map = cv2.copyMakeBorder(seg_map, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=255)
            h, w = img.shape[:2]

        max_x = max(0, w - crop_w)
        max_y = max(0, h - crop_h)

        x = random.randint(0, max_x) if max_x > 0 else 0
        y = random.randint(0, max_y) if max_y > 0 else 0

        for _ in range(10):
            crop_seg = seg_map[y:y + crop_h, x:x + crop_w]
            valid_pixels = crop_seg != 255
            if valid_pixels.any():
                unique, counts = np.unique(crop_seg[valid_pixels], return_counts=True)
                if len(unique) == 0 or (counts.max() / valid_pixels.sum()) <= self.cat_max_ratio:
                    break
            else:
                break

            x = random.randint(0, max_x) if max_x > 0 else 0
            y = random.randint(0, max_y) if max_y > 0 else 0

        data['img'] = img[y:y + crop_h, x:x + crop_w]
        data['gt_semantic_seg'] = seg_map[y:y + crop_h, x:x + crop_w]
        return data


class RandomFlip:
    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, data: Dict) -> Dict:
        if random.random() < self.prob:
            data['img'] = cv2.flip(data['img'], 1)
            data['gt_semantic_seg'] = cv2.flip(data['gt_semantic_seg'], 1)
        return data


class PhotoMetricDistortion:
    def __init__(
        self,
        brightness_delta: int = 32,
        contrast_range: Tuple[float, float] = (0.5, 1.5),
        saturation_range: Tuple[float, float] = (0.5, 1.5),
        hue_delta: int = 18,
    ):
        self.brightness_delta = brightness_delta
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, data: Dict) -> Dict:
        img = data['img'].copy().astype(np.float32)

        if random.random() > 0.5:
            img += random.uniform(-self.brightness_delta, self.brightness_delta)

        if random.random() > 0.5:
            img *= random.uniform(*self.contrast_range)

        img = np.clip(img, 0, 255)
        img_hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)

        if random.random() > 0.5:
            img_hsv[:, :, 1] *= random.uniform(*self.saturation_range)

        if random.random() > 0.5:
            img_hsv[:, :, 0] += random.uniform(-self.hue_delta, self.hue_delta)

        img_hsv[:, :, 0][img_hsv[:, :, 0] > 180] -= 180
        img_hsv[:, :, 0][img_hsv[:, :, 0] < 0] += 180

        data['img'] = cv2.cvtColor(np.clip(img_hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
        return data


class Normalize:
    def __init__(
        self,
        mean: Tuple[float, float, float] = (123.675, 116.28, 103.53),
        std: Tuple[float, float, float] = (58.395, 57.12, 57.375),
        to_rgb: bool = True,
    ):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, data: Dict) -> Dict:
        img = data['img'].astype(np.float32)
        if self.to_rgb and img.ndim == 3 and img.shape[2] == 3:
            img = img[..., ::-1]
        data['img'] = (img - self.mean) / self.std
        return data


class PackSegInputs:
    def __call__(self, data: Dict) -> Dict:
        img = data['img']
        if img.ndim == 3 and img.shape[2] == 3:
            img = img.transpose(2, 0, 1)
        data['img'] = img.astype(np.float32)
        data['gt_semantic_seg'] = data['gt_semantic_seg'].astype(np.int32)
        return data


def build_pipeline(pipeline_config: List[Dict]) -> Compose:
    transforms = []
    for cfg in pipeline_config:
        cfg = cfg.copy()
        transform_type = cfg.pop('type')
        if transform_type in {'LoadImageFromFile', 'LoadAnnotations'}:
            continue
        if transform_type == 'Resize':
            transforms.append(Resize(**cfg))
        elif transform_type == 'RandomCrop':
            transforms.append(RandomCrop(**cfg))
        elif transform_type == 'RandomFlip':
            transforms.append(RandomFlip(**cfg))
        elif transform_type == 'PhotoMetricDistortion':
            transforms.append(PhotoMetricDistortion(**cfg))
        elif transform_type == 'Normalize':
            transforms.append(Normalize(**cfg))
        elif transform_type == 'PackSegInputs':
            transforms.append(PackSegInputs())
        else:
            raise ValueError(f'Unknown transform type: {transform_type}')
    return Compose(transforms)

