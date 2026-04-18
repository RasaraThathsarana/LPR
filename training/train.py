"""
Training script for Swin UPerNet on ADE20K dataset.

This script replicates MMSegmentation's training loop and configuration
while using the standalone ADE20K preprocessing module.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Tuple
import json
import time

from models import build_model
from configs import CONFIG


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
        
        # Setup optimizer and scheduler
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        
        # Setup tensorboard
        self.writer = SummaryWriter(str(self.log_dir))
        
        # Training state
        self.current_iter = 0
        self.current_epoch = 0
        self.best_miou = 0.0
    
    def _build_optimizer(self) -> optim.Optimizer:
        """Build optimizer from config."""
        opt_cfg = self.config['optimizer']
        
        if opt_cfg['type'] == 'SGD':
            return optim.SGD(
                self.model.parameters(),
                lr=opt_cfg['lr'],
                momentum=opt_cfg.get('momentum', 0.9),
                weight_decay=opt_cfg.get('weight_decay', 0.0005)
            )
        elif opt_cfg['type'] == 'AdamW':
            return optim.AdamW(
                self.model.parameters(),
                lr=opt_cfg['lr'],
                betas=opt_cfg.get('betas', (0.9, 0.999)),
                weight_decay=opt_cfg.get('weight_decay', 0.01)
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_cfg['type']}")
    
    def _build_scheduler(self):
        """Build learning rate scheduler from config."""
        # Simple polynomial decay scheduler (matches MMSegmentation)
        total_iters = self.config['train_cfg']['max_iters']
        
        class PolyLR:
            def __init__(self, optimizer, total_iters, power=0.9, eta_min=1e-4):
                self.optimizer = optimizer
                self.total_iters = total_iters
                self.power = power
                self.eta_min = eta_min
                self.base_lr = optimizer.defaults['lr']
            
            def step(self, current_iter):
                lr = self.eta_min + (self.base_lr - self.eta_min) * (
                    (1 - current_iter / self.total_iters) ** self.power
                )
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
        
        return PolyLR(
            self.optimizer,
            total_iters,
            power=self.config['scheduler'].get('power', 0.9),
            eta_min=self.config['scheduler'].get('eta_min', 1e-4)
        )
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        with tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}') as pbar:
            for batch_data in pbar:
                imgs = batch_data['img'].to(self.device)  # (B, 3, H, W)
                segs = batch_data['gt_semantic_seg'].to(self.device)  # (B, H, W)
                
                # Forward pass
                outputs = self.model(imgs)  # (B, num_classes, H, W)
                
                # Compute loss
                loss = self.criterion(outputs, segs)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                # Update scheduler
                self.scheduler.step(self.current_iter)
                
                # Update metrics
                total_loss += loss.item()
                num_batches += 1
                self.current_iter += 1
                
                # Log
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                # Log to tensorboard every log_interval
                if self.current_iter % self.config['log_interval'] == 0:
                    self.writer.add_scalar(
                        'train/loss',
                        loss.item(),
                        self.current_iter
                    )
                    self.writer.add_scalar(
                        'train/lr',
                        self.optimizer.param_groups[0]['lr'],
                        self.current_iter
                    )
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        num_classes = self.config['num_classes']
        hist = np.zeros((num_classes, num_classes))
        
        with tqdm(self.val_loader, desc='Validating') as pbar:
            for batch_data in pbar:
                imgs = batch_data['img'].to(self.device)
                segs = batch_data['gt_semantic_seg'].to(self.device)
                
                # Forward pass
                outputs = self.model(imgs)  # (B, num_classes, H, W)
                
                # Get predictions
                preds = outputs.argmax(dim=1)  # (B, H, W)
                
                # Compute confusion matrix
                for pred, seg in zip(preds, segs):
                    pred = pred.cpu().numpy().flatten()
                    seg = seg.cpu().numpy().flatten()
                    
                    # Ignore label 255
                    mask = seg != 255
                    
                    hist += self._compute_hist(pred[mask], seg[mask], num_classes)
        
        # Compute metrics
        metrics = self._compute_miou(hist)
        
        return metrics
    
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
                iou = 0.0
            else:
                iou = tp / (tp + fp + fn)
            
            ious.append(iou)
        
        miou = np.mean(ious)
        
        return {
            'mIoU': miou,
            'mAcc': np.mean([hist[i, i] / hist[i, :].sum() for i in range(num_classes)]),
            'allAcc': np.trace(hist) / hist.sum(),
        }
    
    def save_checkpoint(self, is_best: bool = False):
        """Save checkpoint."""
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter': self.current_iter,
            'epoch': self.current_epoch,
            'config': self.config,
        }
        
        path = self.checkpoint_dir / f'iter_{self.current_iter}.pth'
        torch.save(checkpoint, str(path))
        
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, str(best_path))
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'], strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.current_iter = checkpoint['iter']
        self.current_epoch = checkpoint['epoch']
    
    def train(self, num_epochs: int, val_interval: int = 1):
        """Train the model."""
        max_iters = self.config['train_cfg']['max_iters']
        
        print(f"Starting training for {num_epochs} epochs ({max_iters} iterations)...")
        
        while self.current_iter < max_iters:
            self.current_epoch += 1
            
            # Train one epoch
            train_loss = self.train_epoch()
            print(f"Epoch {self.current_epoch}: avg loss = {train_loss:.4f}")
            
            # Validate
            if self.current_epoch % val_interval == 0:
                print("Running validation...")
                metrics = self.validate()
                print(f"mIoU: {metrics['mIoU']:.4f}, mAcc: {metrics['mAcc']:.4f}")
                
                # Log metrics
                self.writer.add_scalar('val/mIoU', metrics['mIoU'], self.current_iter)
                self.writer.add_scalar('val/mAcc', metrics['mAcc'], self.current_iter)
                
                # Save checkpoint
                is_best = metrics['mIoU'] > self.best_miou
                if is_best:
                    self.best_miou = metrics['mIoU']
                    self.save_checkpoint(is_best=True)
                else:
                    self.save_checkpoint()
        
        print("Training completed!")
        self.writer.close()


def train(args):
    """Main training function."""
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load configuration
    config = CONFIG[args.config]
    print(f"Config: {args.config}")
    print(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Import data loading utilities
    from datasets import create_train_loader, create_val_loader
    from datasets.ade20k_preprocessing.preprocessing_config import TRAIN_PIPELINE, VAL_PIPELINE
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader = create_train_loader(
        config['data_root'],
        TRAIN_PIPELINE,
        batch_size=config['batch_size']
    )
    val_loader = create_val_loader(
        config['data_root'],
        VAL_PIPELINE,
        batch_size=1
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
    trainer.train(
        num_epochs=config['num_epochs'],
        val_interval=config['val_interval']
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Swin UPerNet on ADE20K')
    parser.add_argument('--config', type=str, default='swin_tiny',
                       choices=['swin_tiny', 'swin_small', 'swin_base', 'swin_large'],
                       help='Model configuration')
    parser.add_argument('--data-root', type=str, default='data/ade/ADEChallengeData2016',
                       help='Path to ADE20K dataset')
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
    
    args = parser.parse_args()
    
    train(args)
