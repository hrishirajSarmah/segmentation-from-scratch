# Image Segmemtation using UNET

A minimal PyTorch + Albumentations implementation of a U-Net for binary image segmentation, with utilities for dataset loading, train/val splits, checkpointing, and saving prediction visualizations.

## Project layout
- `train.py`: training loop, transforms, and hyperparameters.
- `model.py`: U-Net model definition.
- `dataset.py`: `CarvanaDataset` for image/mask pairs.
- `utils.py`: data loaders, checkpoint helpers, metrics, and image saving.
- `split.py`: helper to create validation split directories.
- `data/`: dataset folders (`train_images`, `train_masks`, `val_images`, `val_masks`).
- `saved_images/`: prediction and target previews saved during training.

## Requirements
- Python 3.10+ recommended
- PyTorch, torchvision
- albumentations
- tqdm
- scikit-learn (only for `split.py`)
- Pillow, numpy

If you use conda, activate your environment first.

## Setup
Install dependencies with pip:

```bash
pip install torch torchvision albumentations tqdm scikit-learn pillow numpy
```

On Apple Silicon, install the PyTorch build that supports MPS from the official instructions:

```bash
pip install torch torchvision
```

## Data preparation
Expected directory structure under `data/`:

```
data/
  train_images/
  train_masks/
  val_images/
  val_masks/
```

Mask filenames are expected to follow the pattern used in `dataset.py`:
- image: `abc_01.jpg`
- mask: `abc_01_mask.gif`

### Create a validation split
This moves a percentage of training images/masks into `val_images/` and `val_masks/`.

```bash
python split.py
```

By default it uses a 20% split; change `val_split` in `split.py` if needed.

## Training
Run training with the default hyperparameters in `train.py`:

```bash
python train.py
```

Key defaults (edit in `train.py` if needed):
- `DEVICE`: uses `mps` when available, else `cpu`.
- `IMAGE_HEIGHT` / `IMAGE_WIDTH`: resized input size.
- `BATCH_SIZE`, `NUM_EPOCHS`, `LEARNING_RATE`.
- `TRAIN_IMG_DIR`, `TRAIN_MASK_DIR`, `VAL_IMG_DIR`, `VAL_MASK_DIR`.

### Checkpoints
- Saved to `my_checkpoint.pth.tar` via `save_checkpoint`.
- To resume, set `LOAD_MODEL = True` in `train.py`.

### Prediction previews
During training, prediction previews are saved to `saved_images/`:
- `pred_*.png`: model predictions
- `target_*.png`: ground-truth masks

## Notes
- This project is configured for **binary segmentation** (single output channel). For multi-class segmentation, update `out_channels` and the loss accordingly.
- `split.py` depends on scikit-learn; if you don’t need it, you can skip installing that package.

## Common issues
- If `my_checkpoint.pth.tar` is missing, either set `LOAD_MODEL = False` or place the checkpoint file in the project root.
- If `sklearn` import fails due to SciPy on macOS, reinstall `scikit-learn` and `scipy` in the same environment.
