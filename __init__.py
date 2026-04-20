"""
Swin UPerNet - Semantic Segmentation Training Framework

A standalone implementation of Swin Transformer-based UPerNet for semantic
segmentation on the ADE20K dataset. This framework replicates MMSegmentation's
training setup while providing easy model modifications.

Available models:
- swin_tiny: 29M parameters
- swin_small: 50M parameters
- swin_base: 88M parameters
- swin_large: 197M parameters

Quick start:
    from models import build_model
    from training.train import Trainer
    from configs import CONFIG
    
    # Build model
    model = build_model('swin_tiny')
    
    # Create trainer (requires data loaders)
    trainer = Trainer(model, train_loader, val_loader, CONFIG['swin_tiny'])
    
    # Train
    trainer.train()

For more information, see README.md, SETUP.md, and TRAINING.md.
"""

__version__ = '1.0.0'
__author__ = 'Segmentation Framework'

from .models import EncoderDecoderModel, SegmentationModel, build_model
from .configs import CONFIG
from .evaluation import SegmentationMetrics, evaluate

__all__ = [
    'EncoderDecoderModel',
    'SegmentationModel',
    'build_model',
    'CONFIG',
    'SegmentationMetrics',
    'evaluate',
]
