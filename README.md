# Joint Superpixel and Self-Representation Learning for Scalable Hyperspectral Image Clustering

PyTorch implementation of "Joint Superpixel and Self-Representation Learning for Scalable Hyperspectral Image Clustering".

This project refines differentiable superpixels with an unfolded ADMM-based self-representation module to produce clustering-friendly superpixel features for hyperspectral images (HSI). It builds upon differentiable SLIC ideas popularized by Superpixel Sampling Networks (SSN).

## Features
- Self-representation guidance: unfolded ADMM with learnable thresholds to encourage sparsity and structure.
- Differentiable superpixel assignment (SLIC-like) jointly optimized with representation.
- End-to-end training for HSI subspace clustering with metrics (ACC / Kappa / NMI).

## Requirements
- Python 3.8+
- PyTorch (CUDA optional)
- numpy, matplotlib
- scikit-image, scikit-learn
- torchvision, pillow
- kornia
- tensorboard (optional)

Example install:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # choose your CUDA/CPU build
pip install numpy matplotlib scikit-image scikit-learn pillow kornia tensorboard
```

## Datasets & Paths
The training script expects local .mat files for common HSI datasets. By default, hardcoded paths are used in `train.py`:

- Salinas: `data/salinas/Salinas_corrected.mat`, `data/salinas/Salinas_gt.mat`
- Trento: `data/trento/Trento.mat`, `data/trento/Trento_gt.mat`
- Urban: `data/urban/Urban_corrected.mat`, `data/urban/Urban_gt.mat`

If your dataset is elsewhere, update these paths in `train.py` (see `train()` dataset branches at `code/test/ssn_euclidean_multi_dense_distill/train.py:156`).

## Training
Basic usage:
```
# Salinas
python train.py \
  --dataset salinas \
  --weight_representation 25 \
  --device cuda:0 \
  --log runs/salinas

# Urban
python train.py \
  --dataset urban \
  --weight_representation 70 \
  --device cuda:0 \
  --log runs/urban

# Trento
python train.py \
  --dataset trento \
  --weight_representation 250 \
  --device cuda:0 \
  --log runs/trento
```

Common arguments:
- `--dataset {salinas,trento,urban}`: choose dataset.
- `--train_iter`: total training iterations (default 300).
- `--niter`: differentiable SLIC iterations per forward (default 5).
- `--nspix`: target number of superpixels (auto-set per dataset if not provided).
- `--weight_representation`: weight for representation (reconstruction/sparsity) losses.
- `--weight_noise`: weight for noise regularization.
- `--device`: e.g., `cuda:0` or `cpu`.
- `--log`: directory for logs/TensorBoard.

Notes:
- TensorBoard logs (losses and clustering metrics) are written under a timestamped folder. Launch TensorBoard with `tensorboard --logdir <log_dir>`.
- If you change dataset resolution or target superpixel count, adjust `--nspix` accordingly.

## Evaluation
During training, the script computes subspace clustering metrics on the learned coefficient matrix (C):
- Accuracy (ACC)
- Kappa
- Normalized Mutual Information (NMI)

These are printed to the console and logged to TensorBoard at intervals.

## Trento Dataset Clustering Comparison
Below is a visualization comparing the clustering effects of different methods on the Trento dataset.

![The clustering result of different methods on Trento dataset. (a) False color, (b) Ground truth, (c) K-means, (d) FCM, (e) FINCH, (f) DEKM, (g) SpectralNet, (h) IDEC, (i) MADL, (j) NCSC, (k) Ours](Trento.pdf)

## Acknowledgements
- Superpixel Sampling Networks (SSN): https://arxiv.org/abs/1807.10174
- Original SSN code: https://github.com/NVlabs/ssn_superpixels
- A pure PyTorch differentiable SLIC variant: https://github.com/perrying/diffSLIC
