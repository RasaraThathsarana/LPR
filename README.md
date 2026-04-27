# Swin UPerNet for Semantic Segmentation

A standalone Swin Transformer-based UPerNet implementation for semantic segmentation. It supports ADE20K and Inria Aerial Image Labeling with a shared training loop and easy model modifications.

## Features

✓ **Identical to MMSegmentation**
- Same training configuration and schedule (160k iterations)
- Same augmentation and preprocessing
- Same optimizer settings (AdamW with polynomial LR decay)
- Same evaluation metrics

✓ **Multiple Swin Variants**
- Swin Tiny (29M params)
- Swin Small (50M params)
- Swin Base (88M params)
- Swin Large (197M params)

✓ **Easy to Modify**
- Simple model structure in `model.py`
- Clear configuration in `config.py`
- Customizable training in `train.py`

✓ **Complete Training Pipeline**
- Data loading with preprocessing
- Training with validation
- Metric evaluation
- Checkpoint management
- TensorBoard logging

## Repository Structure

```
swin_upernet_repo/
├── model.py                 # Model architecture
├── train.py                 # Training script
├── inference.py            # Inference/prediction
├── evaluation.py           # Evaluation metrics
├── config.py               # Training configurations
├── requirements.txt        # Dependencies
├── README.md               # This file
├── SETUP.md               # Installation guide
└── TRAINING.md            # Training guide
```

## Installation

### 1. Clone and setup

```bash
cd swin_upernet_repo
pip install -r requirements.txt
```

### 2. Setup ADE20K preprocessing module

```bash
# Copy the ADE20K preprocessing module (if in same workspace)
cp -r ../ade20k_preprocessing ./
```

### 3. Download dataset

```bash
# Download from:
# http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip

# Extract to:
mkdir -p data/ade
unzip ADEChallengeData2016.zip -d data/ade/
```

Alternatively, training and inference can automatically download the dataset if it is missing:

```bash
python train.py --config swin_tiny --download-data
```

### 4. Verify setup

```bash
python -c "from ade20k_preprocessing import ADE20KDataset; print('✓ Setup OK')"
```

## Quick Start

### Training

```bash
# Train Swin Tiny (default, recommended for testing)
python train.py --config swin_tiny

# Train Swin Tiny and download ADE20K automatically if missing
python train.py --config swin_tiny --download-data

# Train Swin Base (slower but better performance)
python train.py --config swin_base

# Train with custom settings
python train.py \
    --config swin_tiny \
    --data-root data/ade/ADEChallengeData2016 \
    --checkpoint-dir checkpoints \
    --log-dir logs

# Train on Inria Aerial Image Labeling dataset
python train.py \
    --config swin_tiny \
    --dataset inria \
    --raw-data-root data/inria/AerialImageDataset
```

### Inference

```bash
# On single image
python inference.py \
    --checkpoint checkpoints/best_model.pth \
    --image path/to/image.jpg \
    --encoder swin_base

# On validation set
python inference.py \
    --checkpoint checkpoints/best_model.pth \
    --dataset-split validation \
    --encoder swin_base
```

### Evaluation

```python
from evaluation import evaluate
from ade20k_preprocessing import create_val_loader
from ade20k_preprocessing.preprocessing_config import VAL_PIPELINE
import torch

# Create loader
val_loader = create_val_loader(
    'data/ade/ADEChallengeData2016',
    VAL_PIPELINE
)

# Load model
model = torch.load('checkpoints/best_model.pth')

# Evaluate
metrics = evaluate(model, val_loader)
print(f"mIoU: {metrics['mIoU']:.4f}")
```

## Training Configuration

All configurations are in `config.py` and match MMSegmentation settings:

| Model | Batch Size | LR | Params | FLOPs |
|-------|------------|-----|--------|-------|
| Swin Tiny | 8 | 6e-5 | 29M | 39G |
| Swin Small | 8 | 6e-5 | 50M | 83G |
| Swin Base | 4 | 6e-5 | 88M | 140G |
| Swin Large | 2 | 6e-5 | 197M | 282G |

### Default Settings

- **Max iterations**: 160,000
- **Validation interval**: 16,000 iterations
- **Optimizer**: AdamW
- **Scheduler**: Polynomial LR decay (power=0.9)
- **Crop size**: 512×512
- **Augmentation**: RandomResize, RandomCrop, RandomFlip, PhotometricDistortion

### Inria Aerial Image Labeling

Use any backbone config with `--dataset inria`:

```bash
python train.py --config swin_base --dataset inria --raw-data-root data/inria/AerialImageDataset
```

For Inria:

- `num_classes` is `2` for background and building
- source images larger than `512x512` are split into `224x224` tiles during dataset preparation after download
- tiled predictions are stitched back together during inference

## Modifying Models

### Change model architecture

Edit `model.py` - the `SimpleSwinUperNet` class:

```python
# Modify backbone
self.backbone = CustomBackbone(...)

# Modify decoder
self.decode_head = CustomDecoder(...)
```

### Add custom layer

```python
class CustomSwinUperNet(SimpleSwinUperNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Add custom layer
        self.custom_layer = nn.Linear(...)
    
    def forward(self, x):
        outputs = super().forward(x)
        outputs = self.custom_layer(outputs)
        return outputs
```

### Change training settings

Edit `config.py`:

```python
SWIN_TINY_CONFIG = {
    ...
    'batch_size': 16,  # Change batch size
    'optimizer': {
        'lr': 1e-4,  # Change learning rate
    },
    ...
}
```

## Preprocessing Details

Data is preprocessed using the ADE20K preprocessing module:

```
Input Image → Load → Reduce Labels → Resize → Crop → Flip → Distort → Normalize → Pack
```

**Key features:**
- Reduce zero label (background → 255)
- Random resize (2048×512 scale, 0.5-2.0 ratio)
- Random crop (512×512)
- Random flip (50% probability)
- Photometric distortion (brightness, contrast, saturation, hue)
- ImageNet normalization

**Output format:**
- Image: (3, 512, 512) float32
- Mask: (512, 512) int32 (values 0-149 or 255)

## Training Output

During training, you'll see:

```
Epoch 1: avg loss = 2.3456
Running validation...
mIoU: 0.3456, mAcc: 0.5678
```

### Monitoring with TensorBoard

```bash
tensorboard --logdir logs
# Open http://localhost:6006
```

Logged metrics:
- Loss curves
- Learning rate schedule
- mIoU, mAcc over time

## Checkpointing

Checkpoints are saved to `checkpoints/` directory:

```
checkpoints/
├── iter_16000.pth    # Checkpoint at 16k iterations
├── iter_32000.pth    # Checkpoint at 32k iterations
├── best_model.pth    # Best model (highest mIoU)
└── ...
```

### Resume training

```bash
python train.py --config swin_tiny --load-from checkpoints/best_model.pth
```

## Expected Performance

Based on MMSegmentation results on ADE20K validation set:

| Model | mIoU | mAcc |
|-------|------|------|
| Swin Tiny | ~42.9 | ~55.2 |
| Swin Small | ~47.6 | ~60.8 |
| Swin Base | ~49.5 | ~62.7 |
| Swin Large | ~50.3 | ~63.5 |

*Note: These are approximate values. Actual results depend on exact training setup and randomness.*

## File Descriptions

### `model.py`
- `SimpleSwinUperNet`: Main model class
- `SegmentationModel`: Base class for models
- `build_model()`: Factory function for creating models

### `train.py`
- `Trainer`: Training loop implementation
- `train()`: Main training function
- Optimizer and scheduler setup

### `inference.py`
- `SegmentationInferencer`: Inference class
- `colorize_pred()`: Visualization utility
- Command-line interface for predictions

### `evaluation.py`
- `SegmentationMetrics`: Metric computation
- `evaluate()`: Evaluation function
- `CheckpointManager`: Checkpoint management

### `config.py`
- Configurations for all Swin variants
- Hyperparameters (LR, batch size, etc.)
- Model architecture details

## Troubleshooting

### Out of Memory (OOM)

Reduce batch size in `config.py`:

```python
SWIN_TINY_CONFIG = {
    'batch_size': 4,  # Reduce from default 8
    ...
}
```

### Training is slow

- Use smaller model (Swin Tiny instead of Base)
- Reduce validation interval
- Use mixed precision training (not yet implemented)

### Validation set not found

Verify `data/ade/ADEChallengeData2016/` has correct structure:

```
data/ade/ADEChallengeData2016/
├── images/
│   ├── training/    (*.jpg)
│   └── validation/   (*.jpg)
└── annotations/
    ├── training/    (*.png)
    └── validation/   (*.png)
```

### Model not loading from checkpoint

Check checkpoint format:

```python
checkpoint = torch.load('checkpoints/best_model.pth')
# Should have 'model', 'optimizer', 'iter' keys
print(checkpoint.keys())
```

## Next Steps

1. **Start training**: `python train.py --config swin_tiny`
2. **Monitor with TensorBoard**: `tensorboard --logdir logs`
3. **Run inference**: `python inference.py --checkpoint checkpoints/best_model.pth --image test.jpg`
4. **Modify model**: Edit `model.py` and `config.py`
5. **Deploy**: Use `inference.py` for predictions

## References

- Paper: ["Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"](https://arxiv.org/abs/2105.01601)
- Repo: [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
- Dataset: [ADE20K](http://groups.csail.mit.edu/vision/datasets/ADE20K/)

## Citation

```bibtex
@inproceedings{liu2021swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutao and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  booktitle={ICCV},
  year={2021}
}

@article{zhou2017scene,
  title={Scene parsing through ade20k dataset},
  author={Zhou, Bolei and Zhao, Hang and Puig, Xavier and Fidler, Sanja and Torralba, Antonio},
  booktitle={CVPR},
  year={2017}
}
```

## License

Apache 2.0 (same as MMSegmentation)
