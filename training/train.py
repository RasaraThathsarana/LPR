"""
Training script for Swin UPerNet on supported semantic segmentation datasets.

This script replicates MMSegmentation's training loop and configuration while
using the standalone dataset preprocessing modules in this repository.
"""

import argparse
import torch
import random
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional
import json
import sys
import datetime

# Ensure the local LPR package root is on PYTHONPATH when running the script directly.
repo_package_root = Path(__file__).resolve().parents[1]
if str(repo_package_root) not in sys.path:
    sys.path.insert(0, str(repo_package_root))

from models import build_model
from models.model import (
    extract_state_dict_from_checkpoint,
    translate_checkpoint_state_dict,
)
from configs import CONFIG
from configs.config import DEFAULT_CONFIG_NAME, build_config
from datasets.ade20k_preprocessing.download import ensure_ade20k_dataset
from datasets.inria_preprocessing.download import ensure_inria_dataset_from_source
from training.losses import CompositeSegmentationLoss


class StdoutLogger:
    """Custom logger to duplicate stdout to a file."""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def isatty(self):
        return self.terminal.isatty()


class Trainer:
    """Trainer class for semantic segmentation."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        config: Dict,
        device: str = 'cuda',
        checkpoint_dir: str = 'checkpoints',
        log_dir: str = 'logs',
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.val_visualization_dir = self.log_dir / 'validation_visualizations'
        self.val_visualization_dir.mkdir(parents=True, exist_ok=True)
        self.val_visualization_samples = 5
        
        # Setup optimizer and scheduler
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        
        # Setup loss function
        loss_cfg = self.config.get('loss', {})
        self.criterion = CompositeSegmentationLoss(
            ignore_index=loss_cfg.get('ignore_index', 255),
            ce_weight=loss_cfg.get('ce_weight', 1.0),
            dice_weight=loss_cfg.get('dice_weight', 1.0),
            boundary_weight=loss_cfg.get('boundary_weight', 1.0),
            dice_smooth=loss_cfg.get('dice_smooth', 1.0),
            boundary_thickness=loss_cfg.get('boundary_thickness', 1),
        )
        self.aux_criterion = CompositeSegmentationLoss(
            ignore_index=loss_cfg.get('ignore_index', 255),
            ce_weight=loss_cfg.get('aux_ce_weight', loss_cfg.get('ce_weight', 1.0)),
            dice_weight=loss_cfg.get('aux_dice_weight', loss_cfg.get('dice_weight', 1.0)),
            boundary_weight=loss_cfg.get('aux_boundary_weight', loss_cfg.get('boundary_weight', 1.0)),
            dice_smooth=loss_cfg.get('dice_smooth', 1.0),
            boundary_thickness=loss_cfg.get('aux_boundary_thickness', loss_cfg.get('boundary_thickness', 1)),
        )
        self.aux_loss_weight = self.config.get('auxiliary_loss_weight', 0.4)
        
        # Setup tensorboard
        self.writer = SummaryWriter(str(self.log_dir))
        
        # Setup AMP scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.device.startswith('cuda'))
        
        # Training state
        self.current_iter = 0
        self.current_epoch = 0
        self.best_miou = 0.0
        
        # Early stopping state
        self.es_cfg = self.config.get('early_stopping', {})
        self.es_enabled = self.es_cfg.get('enabled', False)
        self.es_patience = self.es_cfg.get('patience', 5)
        self.es_min_delta = self.es_cfg.get('min_delta', 0.001)
        self.es_monitor = self.es_cfg.get('monitor', 'loss')
        self.es_mode = self.es_cfg.get('mode', 'min')
        self.es_counter = 0
        self.es_best = float('inf') if self.es_mode == 'min' else float('-inf')
        self.stop_training = False
    
    def _build_optimizer(self) -> optim.Optimizer:
        """Build optimizer from config."""
        opt_cfg = self.config['optimizer']
        trainable_params = [param for param in self.model.parameters() if param.requires_grad]
        
        if opt_cfg['type'] == 'SGD':
            return optim.SGD(
                trainable_params,
                lr=opt_cfg['lr'],
                momentum=opt_cfg.get('momentum', 0.9),
                weight_decay=opt_cfg.get('weight_decay', 0.0005)
            )
        elif opt_cfg['type'] == 'AdamW':
            no_decay_terms = ('absolute_pos_embed', 'relative_position_bias_table', 'norm')
            decay_params = []
            no_decay_params = []
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if any(term in name for term in no_decay_terms):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

            param_groups = [
                {'params': decay_params, 'weight_decay': opt_cfg.get('weight_decay', 0.01)},
                {'params': no_decay_params, 'weight_decay': 0.0},
            ]
            return optim.AdamW(
                param_groups,
                lr=opt_cfg['lr'],
                betas=opt_cfg.get('betas', (0.9, 0.999)),
                weight_decay=opt_cfg.get('weight_decay', 0.01)
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_cfg['type']}")
    
    def _build_scheduler(self):
        """Build learning rate scheduler from config."""
        sched_cfg = self.config.get('scheduler', {})
        sched_type = sched_cfg.get('type', 'poly')
        
        if sched_type == 'plateau':
            class WarmupPlateauLR:
                def __init__(self, optimizer, sched_cfg):
                    self.optimizer = optimizer
                    self.warmup_iters = sched_cfg.get('warmup_iters', 1500)
                    self.warmup_ratio = sched_cfg.get('warmup_ratio', 1e-6)
                    self.base_lrs = [group['lr'] for group in optimizer.param_groups]
                    self.plateau = optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        mode=sched_cfg.get('mode', 'min'),
                        factor=sched_cfg.get('factor', 0.5),
                        patience=sched_cfg.get('patience', 5),
                        min_lr=sched_cfg.get('min_lr', 1e-6),
                    )
                    self.current_iter = 0

                def step(self, current_iter):
                    self.current_iter = current_iter
                    if self.warmup_iters > 0 and current_iter < self.warmup_iters:
                        alpha = current_iter / max(1, self.warmup_iters)
                        factor = self.warmup_ratio + (1.0 - self.warmup_ratio) * alpha
                        for base_lr, param_group in zip(self.base_lrs, self.optimizer.param_groups):
                            param_group['lr'] = base_lr * factor
                    elif self.warmup_iters > 0 and current_iter == self.warmup_iters:
                        for base_lr, param_group in zip(self.base_lrs, self.optimizer.param_groups):
                            param_group['lr'] = base_lr

                def step_metric(self, metric):
                    if self.current_iter >= self.warmup_iters:
                        self.plateau.step(metric)
                        
            return WarmupPlateauLR(self.optimizer, sched_cfg)
            
        total_iters = self.config['train_cfg']['max_iters']
        
        class WarmupPolyLR:
            def __init__(
                self,
                optimizer,
                total_iters,
                warmup_iters=1500,
                warmup_ratio=1e-6,
                power=1.0,
                eta_min=0.0,
            ):
                self.optimizer = optimizer
                self.total_iters = total_iters
                self.warmup_iters = warmup_iters
                self.warmup_ratio = warmup_ratio
                self.power = power
                self.eta_min = eta_min
                self.base_lr = optimizer.defaults['lr']
            
            def step(self, current_iter):
                current_iter = min(current_iter, self.total_iters)
                if self.warmup_iters > 0 and current_iter < self.warmup_iters:
                    alpha = current_iter / max(1, self.warmup_iters)
                    factor = self.warmup_ratio + (1.0 - self.warmup_ratio) * alpha
                    lr = self.base_lr * factor
                else:
                    progress = (current_iter - self.warmup_iters) / max(1, self.total_iters - self.warmup_iters)
                    progress = min(max(progress, 0.0), 1.0)
                    lr = self.eta_min + (self.base_lr - self.eta_min) * ((1 - progress) ** self.power)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
        
        return WarmupPolyLR(
            self.optimizer,
            total_iters,
            warmup_iters=sched_cfg.get('warmup_iters', 1500),
            warmup_ratio=sched_cfg.get('warmup_ratio', 1e-6),
            power=sched_cfg.get('power', 1.0),
            eta_min=sched_cfg.get('eta_min', 0.0)
        )
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        was_training = self.model.training
        self.model.eval()
        
        num_classes = self.config['num_classes']
        hist = np.zeros((num_classes, num_classes))
        total_loss = 0.0
        num_batches = 0
        visual_samples = []
        
        try:
            with tqdm(self.val_loader, desc='Validating') as pbar:
                for batch_data in pbar:
                    imgs = batch_data['img'].to(self.device)
                    segs = batch_data['gt_semantic_seg'].to(self.device)
                    
                    # Forward pass
                    outputs = self.model(imgs)  # (B, num_classes, H, W)
                    loss = self.criterion(outputs, segs).total
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # Get predictions
                    preds = outputs.argmax(dim=1)  # (B, H, W)
                    
                    if len(visual_samples) < self.val_visualization_samples:
                        imgs_np = imgs.cpu().numpy()
                        segs_np = segs.cpu().numpy()
                        preds_np = preds.cpu().numpy()
                        for sample_index in range(imgs_np.shape[0]):
                            if len(visual_samples) >= self.val_visualization_samples:
                                break
                            visual_samples.append((imgs_np[sample_index], segs_np[sample_index], preds_np[sample_index]))

                    if preds.shape == segs.shape:
                        preds_flat = preds.cpu().numpy().flatten()
                        segs_flat = segs.cpu().numpy().flatten()
                        mask = segs_flat != 255
                        hist += self._compute_hist(preds_flat[mask], segs_flat[mask], num_classes)
                    else:
                        # Compute confusion matrix
                        for pred, seg in zip(preds, segs):
                            pred = pred.cpu().numpy()
                            seg = seg.cpu().numpy()
        
                            # Resize prediction to match ground truth shape if needed
                            if pred.shape != seg.shape:
                                if pred.shape[0] >= seg.shape[0] and pred.shape[1] >= seg.shape[1]:
                                    pred = pred[:seg.shape[0], :seg.shape[1]]
                                else:
                                    pred = cv2.resize(
                                        pred.astype(np.uint8),
                                        (seg.shape[1], seg.shape[0]),
                                        interpolation=cv2.INTER_NEAREST
                                    )
        
                            pred = pred.flatten()
                            seg = seg.flatten()
                            
                            # Ignore label 255
                            mask = seg != 255
                            
                            hist += self._compute_hist(pred[mask], seg[mask], num_classes)
            
            # Compute metrics
            metrics = self._compute_miou(hist)
            metrics['iou_per_class'] = self._compute_iou_per_class(hist)
            metrics['loss'] = total_loss / num_batches if num_batches > 0 else 0.0
            if visual_samples:
                self._save_validation_visualizations(visual_samples, self.current_iter)
            
        finally:
            if was_training:
                self.model.train()
                
        return metrics
    
    def _check_early_stopping(self, metrics: Dict[str, float]) -> bool:
        if not self.es_enabled:
            return False
            
        if self.es_monitor not in metrics:
            raise ValueError(f"Early stopping monitor metric '{self.es_monitor}' not found in metrics. Available metrics: {list(metrics.keys())}")
            
        current_metric = metrics[self.es_monitor]
        current_lr = self.optimizer.param_groups[0]['lr']
        
        if not hasattr(self, 'last_lr'):
            self.last_lr = current_lr
            
        if self.es_cfg.get('reset_on_lr_drop', False):
            # Reset counter if LR has decreased
            if current_lr < self.last_lr - 1e-8:
                print(f"Learning rate reduced to {current_lr:.2e}. Resetting early stopping counter.")
                self.last_lr = current_lr
                self.es_counter = 0
            
        if self.es_mode == 'min':
            improved = current_metric < (self.es_best - self.es_min_delta)
        else:
            improved = current_metric > (self.es_best + self.es_min_delta)
            
        if improved:
            self.es_best = current_metric
            self.es_counter = 0
        else:
            self.es_counter += 1
            print(f"Early stopping counter: {self.es_counter} out of {self.es_patience} (at LR {current_lr:.2e})")
            
            if self.es_counter >= self.es_patience:
                if self.es_cfg.get('stop_on_min_lr', False):
                    min_lr = self.config.get('scheduler', {}).get('min_lr', 1e-6)
                    if current_lr <= min_lr + 1e-8:
                        print(f"Learning rate is at minimum ({min_lr}) and no improvement for {self.es_patience} rounds. Stopping training.")
                        return True
                else:
                    print("Early stopping triggered!")
                    return True
                
        return False

    def _compute_hist(self, pred, true, num_classes):
        """Compute confusion matrix."""
        hist = np.bincount(num_classes * true + pred, minlength=num_classes ** 2)
        return hist.reshape(num_classes, num_classes)
    
    def _compute_miou(self, hist) -> Dict[str, float]:
        """Compute mean IoU and other metrics."""
        ious = []
        num_classes = hist.shape[0]
        
        for i in range(num_classes):
            tp = hist[i, i]
            fp = hist[:, i].sum() - tp
            fn = hist[i, :].sum() - tp
            
            if tp + fp + fn == 0:
                ious.append(np.nan)
            else:
                ious.append(tp / (tp + fp + fn))
        
        miou = np.nanmean(ious)
        
        accs = []
        for i in range(num_classes):
            total = hist[i, :].sum()
            if total == 0:
                accs.append(np.nan)
            else:
                accs.append(hist[i, i] / total)

        all_acc = np.trace(hist) / hist.sum() if hist.sum() > 0 else np.nan
        
        return {
            'mIoU': miou,
            'mAcc': np.nanmean(accs),
            'allAcc': all_acc,
        }

    def _compute_iou_per_class(self, hist) -> Dict[int, float]:
        """Compute IoU for each class."""
        num_classes = hist.shape[0]
        ious = {}
        for i in range(num_classes):
            tp = hist[i, i]
            fp = hist[:, i].sum() - tp
            fn = hist[i, :].sum() - tp

            if tp + fp + fn == 0:
                ious[i] = np.nan
            else:
                ious[i] = tp / (tp + fp + fn)

        return ious

    def _image_to_uint8(self, img: np.ndarray) -> np.ndarray:
        if img.dtype == np.float32 or img.dtype == np.float64:
            if img.max() <= 1.0:
                img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        if img.ndim == 3 and img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        return img

    def _get_color_palette(self, num_classes: int):
        palette = []
        for i in range(num_classes):
            if i == 0:
                palette.append((0, 0, 0))
            else:
                palette.append(((37 * i) % 256, (73 * i) % 256, (149 * i) % 256))
        return palette

    def _decode_segmap(self, seg_map: np.ndarray, palette):
        h, w = seg_map.shape
        color_map = np.zeros((h, w, 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_map[seg_map == label] = color
        return color_map

    def _save_validation_visualizations(self, visual_samples, step: int):
        palette = self._get_color_palette(self.config['num_classes'])
        for sample_index, (img, gt, pred) in enumerate(visual_samples, start=1):
            img = self._image_to_uint8(img)
            gt_color = self._decode_segmap(gt, palette)
            pred_color = self._decode_segmap(pred, palette)
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            combined = np.concatenate([img, gt_color, pred_color], axis=1)
            output = Image.fromarray(combined)
            output_path = self.val_visualization_dir / f'val_step_{step:06d}_sample_{sample_index}.png'
            output.save(str(output_path))

    def _update_hist(self, hist: np.ndarray, preds: torch.Tensor, segs: torch.Tensor) -> np.ndarray:
        """Update confusion matrix histogram from model predictions."""
        num_classes = self.config['num_classes']
        if preds.shape == segs.shape:
            preds_np = preds.cpu().numpy().flatten()
            segs_np = segs.cpu().numpy().flatten()
            mask = segs_np != 255
            hist += self._compute_hist(preds_np[mask], segs_np[mask], num_classes)
        else:
            for pred, seg in zip(preds, segs):
                pred = pred.cpu().numpy().flatten()
                seg = seg.cpu().numpy().flatten()
                mask = seg != 255
                hist += self._compute_hist(pred[mask], seg[mask], num_classes)
        return hist
    
    def save_checkpoint(self, is_best: bool = False):
        """Save checkpoint."""
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter': self.current_iter,
            'epoch': self.current_epoch,
            'config': self.config,
            'best_miou': self.best_miou,
            'scaler': self.scaler.state_dict(),
            'es_counter': self.es_counter,
            'es_best': self.es_best,
        }
        if hasattr(self, 'last_lr'):
            checkpoint['last_lr'] = self.last_lr
        
        if hasattr(self.scheduler, 'plateau'):
            checkpoint['scheduler'] = self.scheduler.plateau.state_dict()
            
        path = self.checkpoint_dir / 'last_model.pth'
        torch.save(checkpoint, str(path))
        
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, str(best_path))
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint."""
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

        state_dict = extract_state_dict_from_checkpoint(checkpoint)
        new_state_dict = translate_checkpoint_state_dict(state_dict)
        missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)
        if missing_keys or unexpected_keys:
            print(
                f"Warning: Loaded checkpoint with {len(missing_keys)} missing keys "
                f"and {len(unexpected_keys)} unexpected keys."
            )
        
        if 'optimizer' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            except (ValueError, RuntimeError) as error:
                print(f"Warning: Skipping optimizer state load ({error}).")
        if 'iter' in checkpoint:
            self.current_iter = checkpoint['iter']
        if 'epoch' in checkpoint:
            self.current_epoch = checkpoint['epoch']
        if 'best_miou' in checkpoint:
            self.best_miou = checkpoint['best_miou']
        if 'scaler' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler'])
        if 'es_counter' in checkpoint:
            self.es_counter = checkpoint['es_counter']
        if 'es_best' in checkpoint:
            self.es_best = checkpoint['es_best']
        if 'last_lr' in checkpoint:
            self.last_lr = checkpoint['last_lr']
        if 'scheduler' in checkpoint and hasattr(self.scheduler, 'plateau'):
            self.scheduler.plateau.load_state_dict(checkpoint['scheduler'])
    
    def train(self):
        """Train the model with iteration-based validation."""
        max_iters = self.config['train_cfg']['max_iters']
        val_interval_iters = self.config['train_cfg']['val_interval']
        num_classes = self.config['num_classes']
        accumulation_steps = self.config.get('accumulation_steps', 1)
        
        print(f"Starting training for {max_iters} iterations (validating every {val_interval_iters} iterations)...")
        
        train_hist = np.zeros((num_classes, num_classes))
        
        self.model.train()
        self.optimizer.zero_grad()
        forward_passes = 0
        
        # Track metrics over the accumulation steps for smoother logging
        accum_stats = {
            'loss': 0.0, 'ce': 0.0, 'dice': 0.0, 'boundary': 0.0,
            'aux_ce': 0.0, 'aux_dice': 0.0, 'aux_boundary': 0.0, 'aux_total': 0.0
        }
        
        while self.current_iter < max_iters and not self.stop_training:
            self.current_epoch += 1
            
            with tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}') as pbar:
                for batch_data in pbar:
                    # Stop if we've reached max iterations or early stopped
                    if self.current_iter >= max_iters or self.stop_training:
                        break
                    
                    imgs = batch_data['img'].to(self.device)
                    segs = batch_data['gt_semantic_seg'].to(self.device)
                    
                    # Forward pass with mixed precision
                    autocast_device = 'cuda' if str(self.device).startswith('cuda') else 'cpu'
                    with torch.amp.autocast(autocast_device):
                        outputs, aux_outputs = self.model(imgs, return_aux=True)
                        main_loss = self.criterion(outputs, segs)
                        loss = main_loss.total
                        aux_loss = None
                        if aux_outputs is not None:
                            aux_loss = self.aux_criterion(aux_outputs, segs)
                            loss = loss + self.aux_loss_weight * aux_loss.total

                    # Scale loss for gradient accumulation
                    accum_loss = loss / accumulation_steps

                    # Accumulate metrics for logging (will be averaged over macro-batch)
                    accum_stats['loss'] += loss.item()
                    accum_stats['ce'] += main_loss.ce.item()
                    accum_stats['dice'] += main_loss.dice.item()
                    accum_stats['boundary'] += main_loss.boundary.item()
                    if aux_outputs is not None:
                        accum_stats['aux_ce'] += aux_loss.ce.item()
                        accum_stats['aux_dice'] += aux_loss.dice.item()
                        accum_stats['aux_boundary'] += aux_loss.boundary.item()
                        accum_stats['aux_total'] += (self.aux_loss_weight * aux_loss.total).item()

                    # Backward pass (accumulates gradients)
                    self.scaler.scale(accum_loss).backward()

                    # Update metrics
                    forward_passes += 1
                    train_hist = self._update_hist(train_hist, outputs.detach().argmax(dim=1), segs)

                    # Log
                    pbar.set_postfix({'loss': f'{loss.item():.4f}', 'iter': self.current_iter})
                    
                    # Perform optimization step when accumulation is reached
                    if forward_passes % accumulation_steps == 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                        
                        # Update true iteration count and scheduler
                        self.current_iter += 1
                        if hasattr(self.scheduler, 'step'):
                            self.scheduler.step(self.current_iter)
                        
                        # Log to tensorboard every log_interval
                        if self.current_iter % self.config['log_interval'] == 0:
                            loss_avg = accum_stats['loss'] / accumulation_steps
                            ce_avg = accum_stats['ce'] / accumulation_steps
                            dice_avg = accum_stats['dice'] / accumulation_steps
                            bound_avg = accum_stats['boundary'] / accumulation_steps
                            print(
                                f"\n[Iter {self.current_iter}] "
                                f"Total Loss: {loss_avg:.4f} | "
                                f"Main CE: {ce_avg:.4f} | "
                                f"Main Dice: {dice_avg:.4f} | "
                                f"Main Boundary: {bound_avg:.4f}"
                            )
                            if aux_loss is not None:
                                aux_ce_avg = accum_stats['aux_ce'] / accumulation_steps
                                aux_dice_avg = accum_stats['aux_dice'] / accumulation_steps
                                aux_bound_avg = accum_stats['aux_boundary'] / accumulation_steps
                                aux_total_avg = accum_stats['aux_total'] / accumulation_steps
                                print(
                                    f"[Iter {self.current_iter}] "
                                    f"Aux CE: {aux_ce_avg:.4f} | "
                                    f"Aux Dice: {aux_dice_avg:.4f} | "
                                    f"Aux Boundary: {aux_bound_avg:.4f} | "
                                    f"Aux Weighted: {aux_total_avg:.4f}"
                                )
                            self.writer.add_scalar(
                                'train/loss',
                                loss_avg,
                                self.current_iter
                            )
                            self.writer.add_scalar('train/loss_ce', ce_avg, self.current_iter)
                            self.writer.add_scalar('train/loss_dice', dice_avg, self.current_iter)
                            self.writer.add_scalar('train/loss_boundary', bound_avg, self.current_iter)
                            self.writer.add_scalar(
                                'train/lr',
                                self.optimizer.param_groups[0]['lr'],
                                self.current_iter
                            )
                            
                            # Log training metrics and reset histogram for unbiased logging
                            train_metrics = self._compute_miou(train_hist)
                            self.writer.add_scalar('train/mIoU', train_metrics['mIoU'], self.current_iter)
                            self.writer.add_scalar('train/mAcc', train_metrics['mAcc'], self.current_iter)
                            train_hist = np.zeros((num_classes, num_classes))
                            
                        # Reset stats for the next macro-batch
                        accum_stats = {
                            'loss': 0.0, 'ce': 0.0, 'dice': 0.0, 'boundary': 0.0,
                            'aux_ce': 0.0, 'aux_dice': 0.0, 'aux_boundary': 0.0, 'aux_total': 0.0
                        }
                        
                        # Validate at specified iteration intervals
                        if self.current_iter % val_interval_iters == 0 or self.current_iter >= max_iters:
                            print(f"\n[Iter {self.current_iter}] Running validation...")
                            metrics = self.validate()
                            print(f"[Iter {self.current_iter}] Val mIoU: {metrics['mIoU']:.4f}, Val mAcc: {metrics['mAcc']:.4f}, Val Loss: {metrics['loss']:.4f}\n")
                            
                            # Step plateau scheduler
                            if hasattr(self.scheduler, 'step_metric'):
                                metric_to_track = metrics['loss'] if self.config['scheduler'].get('mode', 'min') == 'min' else metrics['mIoU']
                                self.scheduler.step_metric(metric_to_track)
                            
                            # Log metrics
                            self.writer.add_scalar('val/mIoU', metrics['mIoU'], self.current_iter)
                            self.writer.add_scalar('val/mAcc', metrics['mAcc'], self.current_iter)
                            self.writer.add_scalar('val/loss', metrics['loss'], self.current_iter)
                            self.writer.add_histogram(
                                'val/IoU/per_class',
                                np.array([v for v in metrics['iou_per_class'].values()], dtype=np.float32),
                                self.current_iter
                            )
                            for class_idx, class_iou in metrics['iou_per_class'].items():
                                self.writer.add_scalar(
                                    f'val/IoU/class_{class_idx}',
                                    class_iou,
                                    self.current_iter
                                )
                            
                            # Check early stopping (run before saving so states are correctly updated)
                            if self._check_early_stopping(metrics):
                                self.stop_training = True

                            # Save checkpoint
                            is_best = metrics['mIoU'] > self.best_miou
                            if is_best:
                                self.best_miou = metrics['mIoU']
                                self.save_checkpoint(is_best=True)
                            else:
                                self.save_checkpoint()
                            
                            if self.stop_training:
                                break
        
        print("Training completed!")
        self.writer.close()


def set_random_seed(seed: Optional[int] = None, deterministic: bool = True):
    """Set random seed and deterministic behavior for reproducibility."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Disable non-deterministic CuDNN algorithms for reproducible results.
    # This is important when using seeded training and GPU operations.
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True, warn_only=True)


def worker_init_fn(worker_id):
    """Worker init function to ensure reproducible augmentations in DataLoader."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train(args):
    """Main training function."""
    
    # Setup file logging
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    sys.stdout = StdoutLogger(log_dir / f"train_{timestamp}.log")

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if args.seed is not None or args.deterministic:
        print(
            f"Setting random seed to {args.seed if args.seed is not None else 'None'} "
            "(deterministic: enabled)"
        )
        set_random_seed(args.seed, deterministic=True)
    
    # Load configuration
    config = build_config(args.config, args.dataset)
    if args.data_root:
        config['data_root'] = args.data_root
    if args.raw_data_root:
        config['raw_data_root'] = args.raw_data_root

    print(f"Config: {args.config}")
    print(f"Configuration: {json.dumps(config, indent=2)}")

    dataset_name = config.get('dataset', 'ade20k').lower()

    # Ensure dataset is available
    if dataset_name == 'ade20k':
        ensure_ade20k_dataset(config['data_root'], download=args.download_data)
        from datasets.ade20k_preprocessing.preprocessing_config import (
            VAL_PIPELINE,
            get_train_pipeline,
        )
        TRAIN_PIPELINE = get_train_pipeline(config.get('crop_size'))
    elif dataset_name == 'inria':
        prepared_root = config['data_root']
        raw_root = config.get('raw_data_root', Path(prepared_root).with_name('AerialImageDataset'))
        if args.inria_archive:
            print(f"Preparing the Inria dataset from: {args.inria_archive}")
        elif args.download_data:
            print("Downloading and preparing the Inria dataset...")
        ensure_inria_dataset_from_source(
            raw_root=raw_root,
            prepared_root=prepared_root,
            download=args.download_data,
            archive_path=args.inria_archive,
        )
        from datasets.inria_preprocessing.preprocessing_config import TRAIN_PIPELINE, VAL_PIPELINE
    elif dataset_name == 'pannuke':
        prepared_root = config['data_root']
        from datasets.pannuke_preprocessing.download import ensure_pannuke_dataset
        if args.download_data:
            print('Downloading and preparing the PanNuke dataset from Hugging Face...')
            ensure_pannuke_dataset(prepared_root, download=True)
        else:
            try:
                ensure_pannuke_dataset(prepared_root, download=False)
            except FileNotFoundError:
                from datasets.pannuke_preprocessing.download import download_instructions
                print(download_instructions())
                raise
        from datasets.pannuke_preprocessing.preprocessing_config import TRAIN_PIPELINE, VAL_PIPELINE
    else:
        raise ValueError(f"Unknown dataset in config: {dataset_name}")

    # Import data loading utilities
    from datasets import create_train_loader, create_val_loader
    
    # Dataloader arguments for reproducibility
    dl_kwargs = {}
    if args.seed is not None:
        g = torch.Generator()
        g.manual_seed(args.seed)
        dl_kwargs['worker_init_fn'] = worker_init_fn
        dl_kwargs['generator'] = g
        
    # Create data loaders
    print("Creating data loaders...")
    train_loader = create_train_loader(
        config['data_root'],
        TRAIN_PIPELINE,
        batch_size=config['batch_size'],
        dataset_name=dataset_name,
        split=config.get('train_split', 'training'),
        **dl_kwargs
    )
    val_loader = create_val_loader(
        config['data_root'],
        VAL_PIPELINE,
        batch_size=config.get('val_batch_size') or 1,
        dataset_name=dataset_name,
        split=config.get('val_split', 'validation'),
        **dl_kwargs
    )
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Build model
    print("Building model...")
    model = build_model(
        encoder_name=args.encoder or config['model']['encoder'],
        decoder_name=args.decoder or config['model']['decoder'],
        adapter_name=args.adapter or config['model'].get('adapter', None),
        num_classes=config['num_classes'],
        encoder_kwargs=config['model'].get('encoder_kwargs', {}),
        adapter_kwargs=config['model'].get('adapter_kwargs', {}),
        decoder_kwargs=config['model'].get('decoder_kwargs', {}),
        use_auxiliary_decoder=config['model'].get('use_auxiliary_decoder', True),
        auxiliary_kwargs=config['model'].get('auxiliary_kwargs', {}),
        input_norm_cfg=config.get('data_preprocessor', {}),
        train_encoder=args.train_encoder if args.train_encoder is not None else config['model'].get('train_encoder', True),
        pretrained=config['model'].get('pretrained', False),
        pretrain_path=config['model'].get('pretrain_path', None),
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
    )
    
    # Load checkpoint if specified
    if args.load_from:
        print(f"Loading checkpoint: {args.load_from}")
        trainer.load_checkpoint(args.load_from)
    
    # Train
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Swin UPerNet on a supported segmentation dataset')
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_NAME,
                       choices=list(CONFIG.keys()),
                       help='Model backbone/config name')
    parser.add_argument('--dataset', type=str, default=None,
                       choices=['ade20k', 'inria', 'pannuke'],
                       help='Dataset preset to pair with the selected backbone')
    parser.add_argument('--data-root', type=str, default=None,
                       help='Override prepared dataset root path from config')
    parser.add_argument('--raw-data-root', type=str, default=None,
                       help='Override raw dataset root path for Inria')
    parser.add_argument('--download-data', action='store_true',
                       help='Download and prepare the selected dataset automatically if missing')
    parser.add_argument('--inria-archive', type=str, default=None,
                       help='Path to a local Inria raw folder, archive, or extracted dataset to prepare')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='Directory for tensorboard logs')
    parser.add_argument('--load-from', type=str, default=None,
                       help='Path to checkpoint to load from')
    parser.add_argument('--encoder', type=str, default=None,
                       help='Optional encoder module to override config')
    parser.add_argument('--decoder', type=str, default=None,
                       help='Optional decoder module to override config')
    parser.add_argument('--adapter', type=str, default=None,
                       help='Optional adapter module name to insert between encoder and decoder')
    encoder_train_group = parser.add_mutually_exclusive_group()
    encoder_train_group.add_argument('--train-encoder', dest='train_encoder', action='store_true',
                                     help='Enable encoder training')
    encoder_train_group.add_argument('--freeze-encoder', dest='train_encoder', action='store_false',
                                     help='Disable encoder training')
    parser.set_defaults(train_encoder=None)
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility. When set, deterministic backend options are enabled.')
    parser.add_argument('--deterministic', action='store_true',
                       help='Also enable deterministic backend options when no seed is provided.')
    
    args = parser.parse_args()
    
    train(args)
