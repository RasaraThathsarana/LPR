"""
Evaluation utilities for semantic segmentation.

Provides metrics like mIoU, mAcc, pixel accuracy, etc.
"""

import cv2
import numpy as np
import torch
from typing import Dict


class SegmentationMetrics:
    """Compute segmentation metrics."""
    
    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """Reset metrics."""
        self.hist = np.zeros((self.num_classes, self.num_classes))
    
    def update(self, pred: np.ndarray, gt: np.ndarray):
        """Update metrics with batch.
        
        Args:
            pred: Predictions (H, W) with class indices
            gt: Ground truth (H, W) with class indices
        """
        if pred.shape != gt.shape:
            if pred.shape[0] >= gt.shape[0] and pred.shape[1] >= gt.shape[1]:
                pred = pred[:gt.shape[0], :gt.shape[1]]
            else:
                pred = cv2.resize(
                    pred.astype(np.uint8),
                    (gt.shape[1], gt.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )

        pred = pred.flatten()
        gt = gt.flatten()
        
        # Ignore pixels with ignore_index
        mask = gt != self.ignore_index
        
        pred_valid = pred[mask]
        gt_valid = gt[mask]
        
        # Update confusion matrix
        hist = np.bincount(
            self.num_classes * gt_valid + pred_valid,
            minlength=self.num_classes ** 2
        )
        self.hist += hist.reshape(self.num_classes, self.num_classes)
    
    def compute_miou(self) -> float:
        """Compute mean Intersection over Union."""
        ious = []
        for i in range(self.num_classes):
            tp = self.hist[i, i]
            fp = self.hist[:, i].sum() - tp
            fn = self.hist[i, :].sum() - tp
            
            if tp + fp + fn == 0:
                ious.append(np.nan)
            else:
                ious.append(tp / (tp + fp + fn))
        
        return np.nanmean(ious)
    
    def compute_macc(self) -> float:
        """Compute mean Accuracy per class."""
        accs = []
        for i in range(self.num_classes):
            tp = self.hist[i, i]
            total = self.hist[i, :].sum()
            
            if total == 0:
                accs.append(np.nan)
            else:
                accs.append(tp / total)
        
        return np.nanmean(accs)
    
    def compute_pxa(self) -> float:
        """Compute pixel-wise Accuracy."""
        return np.trace(self.hist) / self.hist.sum()
    
    def compute_all_metrics(self) -> Dict[str, float]:
        """Compute all metrics."""
        return {
            'mIoU': self.compute_miou(),
            'mAcc': self.compute_macc(),
            'pxAcc': self.compute_pxa(),
        }
    
    def compute_iou_per_class(self) -> Dict[int, float]:
        """Compute IoU for each class."""
        ious = {}
        for i in range(self.num_classes):
            tp = self.hist[i, i]
            fp = self.hist[:, i].sum() - tp
            fn = self.hist[i, :].sum() - tp
            
            if tp + fp + fn == 0:
                iou = np.nan
            else:
                iou = tp / (tp + fp + fn)
            
            ious[i] = iou
        
        return ious


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data_loader,
    device: str = 'cuda',
    num_classes: int = 150,
) -> Dict[str, float]:
    """Evaluate model on dataset.
    
    Args:
        model: Segmentation model
        data_loader: Data loader
        device: Device to use
        num_classes: Number of classes
        
    Returns:
        Dictionary with metrics
    """
    model.eval()
    metrics = SegmentationMetrics(num_classes)
    
    for batch_data in data_loader:
        imgs = batch_data['img'].to(device)
        segs = batch_data['gt_semantic_seg'].to(device)
        
        # Forward pass
        outputs = model(imgs)  # (B, num_classes, H, W)
        preds = outputs.argmax(dim=1)  # (B, H, W)
        
        # Update metrics
        for pred, seg in zip(preds, segs):
            metrics.update(pred.cpu().numpy(), seg.cpu().numpy())
    
    return metrics.compute_all_metrics()


if __name__ == '__main__':
    # Example usage
    metrics = SegmentationMetrics(num_classes=150)
    
    # Simulate some predictions and ground truth
    pred = np.random.randint(0, 150, (512, 512))
    gt = np.random.randint(0, 150, (512, 512))
    
    metrics.update(pred, gt)
    
    print("Metrics:")
    for name, value in metrics.compute_all_metrics().items():
        print(f"  {name}: {value:.4f}")
