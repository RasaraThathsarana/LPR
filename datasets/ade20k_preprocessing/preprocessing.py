"""
Preprocessing transforms for ADE20K dataset.
These transforms replicate MMSegmentation preprocessing.
"""
import cv2
import numpy as np
from typing import Dict, Tuple, List
import random


class Compose:
    """Compose multiple transforms."""
    
    def __init__(self, transforms: List['Transform']):
        self.transforms = transforms
    
    def __call__(self, data: Dict) -> Dict:
        for transform in self.transforms:
            data = transform(data)
        return data


class LoadImageFromFile:
    """Load image from file (already done in dataset, but keeping for compatibility)."""
    
    def __call__(self, data: Dict) -> Dict:
        # Image is already loaded by dataset
        return data


class LoadAnnotations:
    """Load annotation from file (already done in dataset, but keeping for compatibility).
    
    Args:
        reduce_zero_label (bool): Whether to reduce zero label. Default: False
    """
    
    def __init__(self, reduce_zero_label: bool = False):
        self.reduce_zero_label = reduce_zero_label
    
    def __call__(self, data: Dict) -> Dict:
        # Segmentation map is already loaded by dataset
        return data


class RandomResize:
    """Randomly resize the image and segmentation map.
    
    Args:
        scale (tuple): Base scale size (W, H)
        ratio_range (tuple): Ratio range for resizing
        keep_ratio (bool): Whether to keep aspect ratio
    """
    
    def __init__(
        self,
        scale: Tuple[int, int],
        ratio_range: Tuple[float, float] = (0.5, 2.0),
        keep_ratio: bool = True
    ):
        self.scale = scale
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
    
    def __call__(self, data: Dict) -> Dict:
        img = data['img']
        seg_map = data['gt_semantic_seg']
        
        # Random ratio
        ratio = random.uniform(*self.ratio_range)
        
        if self.keep_ratio:
            # Scale based on the larger dimension
            h, w = img.shape[:2]
            max_long_edge = max(self.scale)
            max_short_edge = min(self.scale)
            
            scale_factor = min(max_long_edge / max(h, w),
                             max_short_edge / min(h, w)) * ratio
            
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
        else:
            new_w = int(self.scale[0] * ratio)
            new_h = int(self.scale[1] * ratio)
        
        # Resize image
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Resize segmentation map using nearest neighbor
        seg_map = cv2.resize(
            seg_map.astype(np.float32),
            (new_w, new_h),
            interpolation=cv2.INTER_NEAREST
        ).astype(np.int32)
        
        data['img'] = img
        data['gt_semantic_seg'] = seg_map
        return data


class Resize:
    """Resize the image and segmentation map.
    
    Args:
        scale (tuple): Target scale size (W, H)
        keep_ratio (bool): Whether to keep aspect ratio
    """
    
    def __init__(
        self,
        scale: Tuple[int, int],
        keep_ratio: bool = True
    ):
        self.scale = scale
        self.keep_ratio = keep_ratio
    
    def __call__(self, data: Dict) -> Dict:
        img = data['img']
        seg_map = data['gt_semantic_seg']
        
        if self.keep_ratio:
            # Scale based on the larger dimension
            h, w = img.shape[:2]
            max_long_edge = max(self.scale)
            max_short_edge = min(self.scale)
            
            scale_factor = min(max_long_edge / max(h, w),
                             max_short_edge / min(h, w))
            
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
        else:
            new_w = self.scale[0]
            new_h = self.scale[1]
        
        # Resize image
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Resize segmentation map using nearest neighbor
        seg_map = cv2.resize(
            seg_map.astype(np.float32),
            (new_w, new_h),
            interpolation=cv2.INTER_NEAREST
        ).astype(np.int32)
        
        data['img'] = img
        data['gt_semantic_seg'] = seg_map
        return data


class RandomCrop:
    """Randomly crop the image and segmentation map.
    
    Args:
        crop_size (tuple): Size of crop (H, W)
        cat_max_ratio (float): Maximum ratio of category pixels in crop
    """
    
    def __init__(
        self,
        crop_size: Tuple[int, int],
        cat_max_ratio: float = 1.0
    ):
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
    
    def __call__(self, data: Dict) -> Dict:
        img = data['img']
        seg_map = data['gt_semantic_seg']
        
        h, w = img.shape[:2]
        crop_h, crop_w = self.crop_size
        
        if h < crop_h or w < crop_w:
            # Pad if necessary
            pad_h = max(0, crop_h - h)
            pad_w = max(0, crop_w - w)
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w,
                                     cv2.BORDER_REFLECT_101)
            seg_map = cv2.copyMakeBorder(seg_map, 0, pad_h, 0, pad_w,
                                        cv2.BORDER_CONSTANT, value=255)
            h, w = img.shape[:2]
        
        # Random crop position
        max_x = w - crop_w
        max_y = h - crop_h
        
        x = random.randint(0, max_x) if max_x > 0 else 0
        y = random.randint(0, max_y) if max_y > 0 else 0
        
        # Check cat_max_ratio
        for _ in range(10):  # Try up to 10 times
            crop_seg = seg_map[y:y+crop_h, x:x+crop_w]
            # Find ignore label pixels (255)
            valid_pixels = crop_seg != 255
            unique, counts = np.unique(crop_seg[valid_pixels], return_counts=True)
            
            if len(unique) == 0 or (counts.max() / valid_pixels.sum()) <= self.cat_max_ratio:
                break
            
            # Try new position
            x = random.randint(0, max_x) if max_x > 0 else 0
            y = random.randint(0, max_y) if max_y > 0 else 0
        
        img = img[y:y+crop_h, x:x+crop_w]
        seg_map = seg_map[y:y+crop_h, x:x+crop_w]
        
        data['img'] = img
        data['gt_semantic_seg'] = seg_map
        return data


class RandomFlip:
    """Random horizontal flip.
    
    Args:
        prob (float): Probability of flipping
    """
    
    def __init__(self, prob: float = 0.5):
        self.prob = prob
    
    def __call__(self, data: Dict) -> Dict:
        if random.random() < self.prob:
            data['img'] = cv2.flip(data['img'], 1)  # 1 = horizontal flip
            data['gt_semantic_seg'] = cv2.flip(data['gt_semantic_seg'], 1)
        return data


class PhotoMetricDistortion:
    """Apply photometric distortions to image.
    
    Randomly adjust brightness, contrast, saturation, and hue.
    """
    
    def __init__(
        self,
        brightness_delta: int = 32,
        contrast_range: Tuple[float, float] = (0.5, 1.5),
        saturation_range: Tuple[float, float] = (0.5, 1.5),
        hue_delta: int = 18
    ):
        self.brightness_delta = brightness_delta
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_delta = hue_delta
    
    def __call__(self, data: Dict) -> Dict:
        img = data['img'].copy().astype(np.float32)
        
        # Random brightness
        if random.random() > 0.5:
            delta = random.uniform(-self.brightness_delta, self.brightness_delta)
            img += delta
        
        # Random contrast
        if random.random() > 0.5:
            alpha = random.uniform(*self.contrast_range)
            img *= alpha
        
        img = np.clip(img, 0, 255)
        
        # Convert to HSV for saturation and hue adjustment
        img_hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Random saturation
        if random.random() > 0.5:
            alpha = random.uniform(*self.saturation_range)
            img_hsv[:, :, 1] *= alpha
        
        # Random hue
        if random.random() > 0.5:
            delta = random.uniform(-self.hue_delta, self.hue_delta)
            img_hsv[:, :, 0] += delta
        
        img_hsv[:, :, 0][img_hsv[:, :, 0] > 180] -= 180
        img_hsv[:, :, 0][img_hsv[:, :, 0] < 0] += 180
        
        img_hsv = np.clip(img_hsv, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        
        data['img'] = img
        return data


class Normalize:
    """Normalize image with mean and std.
    
    Args:
        mean (tuple): Mean values for RGB channels
        std (tuple): Standard deviation for RGB channels
        to_rgb (bool): Whether to convert BGR to RGB
    """
    
    def __init__(
        self,
        mean: Tuple[float, float, float] = (123.675, 116.28, 103.53),
        std: Tuple[float, float, float] = (58.395, 57.12, 57.375),
        to_rgb: bool = True
    ):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb
    
    def __call__(self, data: Dict) -> Dict:
        img = data['img'].astype(np.float32)
        
        if self.to_rgb and img.shape[2] == 3:
            # Convert BGR to RGB (PIL loads as RGB, but just in case)
            img = img[..., ::-1]
        
        img = (img - self.mean) / self.std
        
        data['img'] = img
        return data


class PackSegInputs:
    """Pack segmentation inputs for model.
    
    Convert to standard format:
    - img: float32, shape (C, H, W)
    - gt_semantic_seg: int32, shape (H, W)
    """
    
    def __call__(self, data: Dict) -> Dict:
        img = data['img']
        
        # Convert to CHW format
        if img.ndim == 3 and img.shape[2] == 3:
            img = img.transpose(2, 0, 1)
        
        data['img'] = img.astype(np.float32)
        data['gt_semantic_seg'] = data['gt_semantic_seg'].astype(np.int32)
        
        return data


def build_pipeline(pipeline_config: List[Dict]) -> Compose:
    """Build a preprocessing pipeline from config.
    
    Args:
        pipeline_config: List of transform configs
        
    Returns:
        Compose object with transforms
    """
    transforms = []
    
    for cfg in pipeline_config:
        cfg = cfg.copy()
        transform_type = cfg.pop('type')
        
        if transform_type == 'LoadImageFromFile':
            transforms.append(LoadImageFromFile())
        elif transform_type == 'LoadAnnotations':
            transforms.append(LoadAnnotations(**cfg))
        elif transform_type == 'RandomResize':
            transforms.append(RandomResize(**cfg))
        elif transform_type == 'Resize':
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
            raise ValueError(f"Unknown transform type: {transform_type}")
    
    return Compose(transforms)
