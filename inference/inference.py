"""
Inference script for Swin UPerNet semantic segmentation.

Create predictions on images or datasets.
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import cv2

from models import build_model
from models.model import translate_checkpoint_state_dict
from datasets import ADE20KDataset, InriaAerialImageDataset
from datasets.ade20k_preprocessing.download import ensure_ade20k_dataset
from datasets.inria_preprocessing.download import ensure_inria_dataset
from datasets.ade20k_preprocessing.preprocessing import build_pipeline
from datasets.ade20k_preprocessing.preprocessing_config import VAL_PIPELINE as ADE20K_VAL_PIPELINE
from datasets.inria_preprocessing.preprocessing_config import VAL_PIPELINE as INRIA_VAL_PIPELINE
from datasets.inria_preprocessing.inria_dataset import build_tile_coordinates, stitch_tile_logits
from evaluation.evaluation import SegmentationMetrics
from configs import CONFIG


class SegmentationInferencer:
    """Perform inference on images."""
    
    def __init__(
        self,
        checkpoint_path: str,
        encoder_name: str = 'swin_base',
        decoder_name: str = 'upernet',
        adapter_name: str = None,
        num_classes: int = 150,
        encoder_kwargs: dict = None,
        decoder_kwargs: dict = None,
        use_auxiliary_decoder: bool = True,
        auxiliary_kwargs: dict = None,
        input_norm_cfg: dict = None,
        device: str = 'cuda',
    ):
        self.device = device
        self.num_classes = num_classes
        self.dataset_name = None
        self.tile_size = 224
        self.large_image_threshold = 512
        
        # Build identical preprocessing pipeline to training
        if input_norm_cfg is None:
            input_norm_cfg = {}
        self.input_norm_cfg = input_norm_cfg
        
        # Build model
        self.model = build_model(
            encoder_name=encoder_name,
            decoder_name=decoder_name,
            adapter_name=adapter_name,
            num_classes=num_classes,
            encoder_kwargs=encoder_kwargs,
            decoder_kwargs=decoder_kwargs,
            use_auxiliary_decoder=use_auxiliary_decoder,
            auxiliary_kwargs=auxiliary_kwargs,
            input_norm_cfg=input_norm_cfg,
            pretrained=False,
        ).to(device)

        self.dataset_name = input_norm_cfg.get('dataset', 'ade20k')
        if self.dataset_name == 'inria':
            self.pipeline = build_pipeline(INRIA_VAL_PIPELINE)
            self.tile_size = 224
            self.large_image_threshold = 512
        else:
            self.pipeline = build_pipeline(ADE20K_VAL_PIPELINE)
        
        # Load checkpoint
        self._load_checkpoint(checkpoint_path)
        self.model.eval()
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 1. Look for 'state_dict' which MMSegmentation uses
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        new_state_dict = translate_checkpoint_state_dict(state_dict)
        missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)
        
        if missing_keys or unexpected_keys:
            print(f"Warning: Loaded with {len(missing_keys)} missing keys and {len(unexpected_keys)} unexpected keys.")
    
    @torch.no_grad()
    def _predict_logits_from_array(self, image: np.ndarray) -> np.ndarray:
        """Run the model on a single processed image/patch and return logits."""
        sample = {
            'img': image,
            'gt_semantic_seg': np.zeros((image.shape[0], image.shape[1]), dtype=np.int32),
        }
        processed = self.pipeline(sample)
        img_tensor = torch.from_numpy(processed['img']).unsqueeze(0).to(self.device)
        output = self.model(img_tensor)
        return output[0].cpu().numpy()

    @torch.no_grad()
    def infer_image(
        self,
        image_path: str,
    ) -> np.ndarray:
        """Infer on a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Segmentation map (H, W) with class indices
        """
        # Load image natively as RGB
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        if self.dataset_name == 'inria' and max(height, width) > self.large_image_threshold:
            tile_coords = build_tile_coordinates(
                height=height,
                width=width,
                tile_size=self.tile_size,
                threshold=self.large_image_threshold,
            )
            tile_logits = []
            for top, left in tile_coords:
                bottom = min(top + self.tile_size, height)
                right = min(left + self.tile_size, width)
                tile = image[top:bottom, left:right]
                if tile.shape[0] < self.tile_size or tile.shape[1] < self.tile_size:
                    tile = cv2.copyMakeBorder(
                        tile,
                        0,
                        self.tile_size - tile.shape[0],
                        0,
                        self.tile_size - tile.shape[1],
                        cv2.BORDER_REFLECT_101,
                    )
                logits = self._predict_logits_from_array(tile)
                tile_logits.append(logits)

            return stitch_tile_logits(
                tile_logits=tile_logits,
                tile_coords=tile_coords,
                image_shape=(height, width),
                tile_size=self.tile_size,
            )

        logits = self._predict_logits_from_array(image)
        pred = logits.argmax(axis=0).astype(np.uint8)
        if pred.shape != (height, width):
            pred = cv2.resize(pred, (width, height), interpolation=cv2.INTER_NEAREST)
        return pred
    
    @torch.no_grad()
    def infer_dataset(
        self,
        dataset,
    ):
        """Infer on a dataset and compute metrics."""
        predictions = []
        metrics = SegmentationMetrics(self.num_classes)
        
        for idx in range(len(dataset)):
            print(f"Processing {idx+1}/{len(dataset)}...")
            
            sample = dataset[idx]
            original_gt = sample.get('gt_semantic_seg')
            if original_gt is not None:
                original_gt = original_gt.copy()
            
            processed = self.pipeline(sample)
            img_tensor = torch.from_numpy(processed['img']).unsqueeze(0).to(self.device)
            output = self.model(img_tensor)
            pred = output.argmax(dim=1)[0].cpu().numpy()
            predictions.append(pred)
            
            if original_gt is not None:
                # The pipeline evaluates against the original unresized ground truth size
                if pred.shape != original_gt.shape:
                    if pred.shape[0] >= original_gt.shape[0] and pred.shape[1] >= original_gt.shape[1]:
                        pred = pred[:original_gt.shape[0], :original_gt.shape[1]]
                    else:
                        pred = cv2.resize(pred.astype(np.uint8), (original_gt.shape[1], original_gt.shape[0]), interpolation=cv2.INTER_NEAREST)
                metrics.update(pred, original_gt)
        
        return {
            'predictions': predictions,
            'metrics': metrics.compute_all_metrics() if len(dataset) > 0 else {},
        }


def colorize_pred(pred: np.ndarray, palette) -> np.ndarray:
    """Colorize prediction with palette.
    
    Args:
        pred: Prediction (H, W) with class indices
        palette: Color palette (num_classes, 3) with RGB values
        
    Returns:
        Colored image (H, W, 3)
    """
    colored = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    
    for class_id, color in enumerate(palette):
        mask = pred == class_id
        colored[mask] = color
    
    return colored


def visualize_predictions(
    image_path: str,
    prediction_path: str,
    output_path: str,
    palette,
    alpha: float = 0.5,
):
    """Visualize predictions alongside original image.
    
    Args:
        image_path: Path to original image
        prediction_path: Path to prediction file
        output_path: Output path for visualization
        palette: Color palette
        alpha: Blending alpha
    """
    # Load image
    image = Image.open(image_path)
    image_array = np.array(image)
    
    # Load prediction
    pred = np.load(prediction_path)
    
    # Colorize prediction
    pred_colored = colorize_pred(pred, palette)
    
    # Blend
    blended = cv2.addWeighted(
        image_array,
        1 - alpha,
        pred_colored,
        alpha,
        0
    )
    
    # Save
    Image.fromarray(blended.astype(np.uint8)).save(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Infer with Swin UPerNet')
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to checkpoint'
    )
    parser.add_argument(
        '--image', type=str, default=None,
        help='Path to image file'
    )
    parser.add_argument(
        '--dataset-split', type=str, default=None,
        choices=['training', 'validation'],
        help='Dataset split to infer on'
    )
    parser.add_argument(
        '--data-root', type=str, default='data/ade/ADEChallengeData2016',
        help='Path to dataset root'
    )
    parser.add_argument(
        '--download-data', action='store_true',
        help='Download ADE20K dataset automatically if missing'
    )
    parser.add_argument(
        '--output-dir', type=str, default='predictions',
        help='Output directory for predictions'
    )
    parser.add_argument(
        '--encoder', type=str, default='swin_base',
        help='Encoder module name'
    )
    parser.add_argument(
        '--decoder', type=str, default='upernet',
        help='Decoder module name'
    )
    parser.add_argument(
        '--adapter', type=str, default=None,
        help='Optional adapter module name'
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Device to use'
    )
    
    args = parser.parse_args()
    
    # Smart configuration loading mapped from the --encoder string
    if args.encoder in CONFIG:
        cfg = CONFIG[args.encoder]
        encoder_name = cfg['model']['encoder']
        encoder_kwargs = cfg['model'].get('encoder_kwargs', {})
        decoder_kwargs = cfg['model'].get('decoder_kwargs', {})
        use_auxiliary_decoder = cfg['model'].get('use_auxiliary_decoder', True)
        auxiliary_kwargs = cfg['model'].get('auxiliary_kwargs', {})
        input_norm_cfg = cfg.get('data_preprocessor', {})
        input_norm_cfg = {**input_norm_cfg, 'dataset': cfg.get('dataset', 'ade20k')}
    else:
        encoder_name = args.encoder
        encoder_kwargs = {}
        decoder_kwargs = {}
        use_auxiliary_decoder = True
        auxiliary_kwargs = {}
        input_norm_cfg = {}
    
    # Create inferencer
    inferencer = SegmentationInferencer(
        checkpoint_path=args.checkpoint,
        encoder_name=encoder_name,
        decoder_name=args.decoder,
        adapter_name=args.adapter,
        encoder_kwargs=encoder_kwargs,
        decoder_kwargs=decoder_kwargs,
        use_auxiliary_decoder=use_auxiliary_decoder,
        auxiliary_kwargs=auxiliary_kwargs,
        input_norm_cfg=input_norm_cfg,
        device=args.device,
    )
    
    # Get palette for visualization
    dataset_name = cfg.get('dataset', 'ade20k') if args.encoder in CONFIG else 'ade20k'
    palette = InriaAerialImageDataset.get_palette() if dataset_name == 'inria' else ADE20KDataset.get_palette()
    
    if args.image:
        # Infer on single image
        print(f"Inferring on {args.image}...")
        pred = inferencer.infer_image(args.image)
        
        # Save prediction
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        pred_path = output_dir / (Path(args.image).stem + '_pred.npy')
        np.save(str(pred_path), pred)
        
        # Visualize
        viz_path = output_dir / (Path(args.image).stem + '_colored.png')
        colored = colorize_pred(pred, palette)
        Image.fromarray(colored).save(str(viz_path))
        
        print(f"Saved prediction to {pred_path}")
        print(f"Saved visualization to {viz_path}")
    
    elif args.dataset_split:
        # Infer on dataset
        print(f"Inferring on {args.dataset_split} split...")
        
        if dataset_name == 'inria':
            ensure_inria_dataset(args.data_root, download=args.download_data)
            dataset = InriaAerialImageDataset(
                data_root=args.data_root,
                split=args.dataset_split,
            )
        else:
            ensure_ade20k_dataset(args.data_root, download=args.download_data)
            dataset = ADE20KDataset(
                data_root=args.data_root,
                split=args.dataset_split,
                reduce_zero_label=True,
            )
        
        results = inferencer.infer_dataset(dataset)
        predictions = results['predictions']
        metrics = results['metrics']
        
        # Save predictions and metrics
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        prediction_files = []
        for idx, pred in enumerate(predictions):
            pred_path = output_dir / f'pred_{idx:06d}.npy'
            np.save(str(pred_path), pred)
            prediction_files.append(str(pred_path.name))
        
        metrics_path = output_dir / f'metrics_{args.dataset_split}.json'
        with open(metrics_path, 'w', encoding='utf-8') as metrics_file:
            json.dump(metrics, metrics_file, indent=2)
        
        summary = {
            'split': args.dataset_split,
            'num_predictions': len(predictions),
            'prediction_files': prediction_files,
            'metrics': metrics,
        }
        summary_path = output_dir / f'summary_{args.dataset_split}.json'
        with open(summary_path, 'w', encoding='utf-8') as summary_file:
            json.dump(summary, summary_file, indent=2)
        
        print(f"Saved {len(predictions)} predictions to {output_dir}")
        print(f"Validation metrics saved to {metrics_path}")
        print(f"Summary saved to {summary_path}")
        print("Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
