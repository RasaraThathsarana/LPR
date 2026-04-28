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
from pathlib import Path
from tqdm import tqdm
from typing import Dict
import json

from models import build_model
from models.model import translate_checkpoint_state_dict
from configs import CONFIG
from configs.config import DEFAULT_CONFIG_NAME, build_config
from datasets.ade20k_preprocessing.download import ensure_ade20k_dataset
from datasets.inria_preprocessing.download import ensure_inria_dataset_from_source
from training.losses import CompositeSegmentationLoss


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
        loss_cfg = self.config.get('loss', {})
        self.criterion = CompositeSegmentationLoss(
            ignore_index=loss_cfg.get('ignore_index', 255),
            ce_weight=loss_cfg.get('ce_weight', 1.0),
            dice_weight=loss_cfg.get('dice_weight', 1.0),
            boundary_weight=loss_cfg.get('boundary_weight', 1.0),
            dice_smooth=loss_cfg.get('dice_smooth', 1.0),
        )
        self.aux_criterion = CompositeSegmentationLoss(
            ignore_index=loss_cfg.get('ignore_index', 255),
            ce_weight=loss_cfg.get('aux_ce_weight', loss_cfg.get('ce_weight', 1.0)),
            dice_weight=loss_cfg.get('aux_dice_weight', loss_cfg.get('dice_weight', 1.0)),
            boundary_weight=loss_cfg.get('aux_boundary_weight', loss_cfg.get('boundary_weight', 1.0)),
            dice_smooth=loss_cfg.get('dice_smooth', 1.0),
        )
        self.aux_loss_weight = self.config.get('auxiliary_loss_weight', 0.4)
        
        # Setup tensorboard
        self.writer = SummaryWriter(str(self.log_dir))
        
        # Setup AMP scaler for mixed precision
        self.scaler = torch.amp.GradScaler(enabled=self.device.startswith('cuda'))
        
        # Training state
        self.current_iter = 0
        self.current_epoch = 0
        self.best_miou = 0.0
    
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
        total_iters = self.config['train_cfg']['max_iters']
        sched_cfg = self.config.get('scheduler', {})
        
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

    def _update_hist(self, hist: np.ndarray, preds: torch.Tensor, segs: torch.Tensor) -> np.ndarray:
        """Update confusion matrix histogram from model predictions."""
        num_classes = self.config['num_classes']
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
        }
        
        path = self.checkpoint_dir / 'last_model.pth'
        torch.save(checkpoint, str(path))
        
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, str(best_path))
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
        new_state_dict = translate_checkpoint_state_dict(state_dict)
        self.model.load_state_dict(new_state_dict, strict=False)
        
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
        
        while self.current_iter < max_iters:
            self.current_epoch += 1
            
            with tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}') as pbar:
                for batch_data in pbar:
                    # Stop if we've reached max iterations
                    if self.current_iter >= max_iters:
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

                    # Backward pass (accumulates gradients)
                    self.scaler.scale(accum_loss).backward()

                    # Update metrics
                    forward_passes += 1
                    train_hist = self._update_hist(train_hist, outputs.argmax(dim=1), segs)

                    # Log
                    pbar.set_postfix({'loss': f'{loss.item():.4f}', 'iter': self.current_iter})
                    
                    # Perform optimization step when accumulation is reached
                    if forward_passes % accumulation_steps == 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                        
                        # Update scheduler and true iteration count
                        self.scheduler.step(self.current_iter)
                        self.current_iter += 1
                        
                        # Log to tensorboard every log_interval
                        if self.current_iter % self.config['log_interval'] == 0:
                            print(
                                f"\n[Iter {self.current_iter}] "
                                f"Total Loss: {loss.item():.4f} | "
                                f"Main CE: {main_loss.ce.item():.4f} | "
                                f"Main Dice: {main_loss.dice.item():.4f} | "
                                f"Main Boundary: {main_loss.boundary.item():.4f}"
                            )
                            if aux_loss is not None:
                                print(
                                    f"[Iter {self.current_iter}] "
                                    f"Aux CE: {aux_loss.ce.item():.4f} | "
                                    f"Aux Dice: {aux_loss.dice.item():.4f} | "
                                    f"Aux Boundary: {aux_loss.boundary.item():.4f} | "
                                    f"Aux Weighted: {(self.aux_loss_weight * aux_loss.total).item():.4f}"
                                )
                            self.writer.add_scalar(
                                'train/loss',
                                loss.item(),
                                self.current_iter
                            )
                            self.writer.add_scalar('train/loss_ce', main_loss.ce.item(), self.current_iter)
                            self.writer.add_scalar('train/loss_dice', main_loss.dice.item(), self.current_iter)
                            self.writer.add_scalar('train/loss_boundary', main_loss.boundary.item(), self.current_iter)
                            self.writer.add_scalar(
                                'train/lr',
                                self.optimizer.param_groups[0]['lr'],
                                self.current_iter
                            )
                        
                        # Validate at specified iteration intervals
                        if self.current_iter % val_interval_iters == 0 or self.current_iter >= max_iters:
                            train_metrics = self._compute_miou(train_hist)
                            print(f"\n[Iter {self.current_iter}] Train mIoU: {train_metrics['mIoU']:.4f}, Train mAcc: {train_metrics['mAcc']:.4f}")
                            
                            print(f"[Iter {self.current_iter}] Running validation...")
                            metrics = self.validate()
                            print(f"[Iter {self.current_iter}] Val mIoU: {metrics['mIoU']:.4f}, Val mAcc: {metrics['mAcc']:.4f}\n")
                            
                            # Log metrics
                            self.writer.add_scalar('train/mIoU', train_metrics['mIoU'], self.current_iter)
                            self.writer.add_scalar('train/mAcc', train_metrics['mAcc'], self.current_iter)
                            self.writer.add_scalar('val/mIoU', metrics['mIoU'], self.current_iter)
                            self.writer.add_scalar('val/mAcc', metrics['mAcc'], self.current_iter)
                            
                            # Save checkpoint
                            is_best = metrics['mIoU'] > self.best_miou
                            if is_best:
                                self.best_miou = metrics['mIoU']
                                self.save_checkpoint(is_best=True)
                            else:
                                self.save_checkpoint()
                            
                            # Reset training histogram after logging
                            train_hist = np.zeros((num_classes, num_classes))
                            
                            # Resume training mode
                            self.model.train()
        
        print("Training completed!")
        self.writer.close()

def set_random_seed(seed: int, deterministic: bool = False):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if args.seed is not None:
        print(f"Setting random seed to {args.seed} (deterministic: {args.deterministic})")
        set_random_seed(args.seed, args.deterministic)
    
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
        from datasets.ade20k_preprocessing.preprocessing_config import TRAIN_PIPELINE, VAL_PIPELINE
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
    else:
        raise ValueError(f"Unknown dataset in config: {dataset_name}")

    # Import data loading utilities
    from datasets import create_train_loader, create_val_loader
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader = create_train_loader(
        config['data_root'],
        TRAIN_PIPELINE,
        batch_size=config['batch_size'],
        dataset_name=dataset_name,
    )
    val_loader = create_val_loader(
        config['data_root'],
        VAL_PIPELINE,
        batch_size=config['batch_size'],
        dataset_name=dataset_name,
    )
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    if args.seed is not None:
        # Ensure reproducibility in multiprocessing workers
        train_loader.worker_init_fn = worker_init_fn
        val_loader.worker_init_fn = worker_init_fn
        g = torch.Generator()
        g.manual_seed(args.seed)
        train_loader.generator = g
        val_loader.generator = g
    
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
                       choices=['ade20k', 'inria'],
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
                       help='Random seed for reproducibility')
    parser.add_argument('--deterministic', action='store_true',
                       help='Whether to set deterministic options for CUDNN backend')
    
    args = parser.parse_args()
    
    train(args)
