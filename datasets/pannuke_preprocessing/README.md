PanNuke preparation and training
================================

Steps to prepare and train on PanNuke:

1. Download PanNuke raw data (manual):
   - Place images under `data/pannuke_raw/images/<split>` and instance annotations under `data/pannuke_raw/instances/<split>`.
2. Prepare dataset (convert instances -> semantic masks):
```bash
python tools/pannuke_prepare.py --raw-root data/pannuke_raw --out-root data/pannuke
```
3. Train using a Swin backbone (example):
```bash
python training/train.py --config swin_base --dataset pannuke --data-root data/pannuke --checkpoint-dir checkpoints/pannuke
```

Notes:
- PanNuke images are 256x256; `crop_size` and pipelines use 256.
- Automatic download is not implemented due to dataset access restrictions; see `download_instructions()`.
