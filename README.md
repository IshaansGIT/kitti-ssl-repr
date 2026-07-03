# Unsupervised Representation Learning on KITTI Raw

SimCLR-based self-supervised pretraining on unlabeled KITTI Raw frames, fine-tuned
on a small labeled scene-classification subset, compared against training from
scratch, with t-SNE/PCA embedding visualization.

## Repo layout

```
configs/
    config.yaml            # all paths, hyperparameters, sequence->category mapping
data/
    raw/                    # KITTI zips/folders go here (gitignored)
    manifests/              # generated CSV manifests (unlabeled + labeled splits)
src/
    data/
        kitti_dataset.py     # SimCLR-style unlabeled dataset (returns 2 augmented views)
        labeled_dataset.py   # labeled scene-classification dataset
        prepare_data.py      # scans raw KITTI folders -> writes manifests
    models/
        backbone.py           # ResNet-18 encoder loader
        simclr.py              # SimCLR model = backbone + projection head
    losses/
        nt_xent.py            # NT-Xent contrastive loss
    train/
        pretrain_simclr.py     # Stage 1: self-supervised pretraining
        finetune.py             # Stage 2a: fine-tune pretrained backbone
        train_scratch.py        # Stage 2b: train identical arch from scratch
    eval/
        extract_embeddings.py  # dump embeddings for a dataset
        visualize_embeddings.py # t-SNE / PCA plots
        clustering_metrics.py   # silhouette, Davies-Bouldin
    utils/
        seed.py
        config.py
notebooks/
    colab_run.ipynb          # thin runner: clones/pulls repo, calls src/ modules on GPU
outputs/
    checkpoints/             # saved model weights (gitignored, large)
    figures/                 # t-SNE/PCA plots, loss curves
```

## Why this structure

All real logic lives in `src/` as plain importable Python — nothing is trapped
inside a notebook. `notebooks/colab_run.ipynb` is intentionally thin: it just
installs deps and calls into `src/train/*.py`, so the same code runs
identically locally (CPU, for debugging on tiny subsets) and on Colab (GPU,
for real runs).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate       # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Getting KITTI data

1. Register and download from https://www.cvlibs.net/datasets/kitti/raw_data.php
2. Pick a handful of sequences from 2-3 of the category tabs (City / Residential
   / Road / Campus) — a few hundred frames total is plenty (we only need ≤500).
3. Download the **synced+rectified** data for `image_02` (left color camera) for
   each sequence you pick. Extract into `data/raw/` so you end up with:
   ```
   data/raw/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/*.png
   data/raw/2011_09_26/2011_09_26_drive_0005_sync/image_02/data/*.png
   ...
   ```
4. Edit `configs/config.yaml` -> `sequences:` and record which category tab
   each drive came from (this is the only reliable source of the category
   label — KITTI does not encode it in the folder name itself).
5. Run:
   ```bash
   python -m src.data.prepare_data --config configs/config.yaml
   ```
   This writes `data/manifests/unlabeled.csv` and `data/manifests/labeled.csv`.

## Pipeline

```bash
# Stage 1: self-supervised pretraining (SimCLR) on unlabeled.csv
python -m src.train.pretrain_simclr --config configs/config.yaml

# Stage 2a: fine-tune the pretrained encoder on labeled.csv
python -m src.train.finetune --config configs/config.yaml

# Stage 2b: baseline, same architecture, random init, trained on labeled.csv
python -m src.train.train_scratch --config configs/config.yaml

# Stage 3: embeddings + visualization + clustering metrics
python -m src.eval.extract_embeddings --config configs/config.yaml --model pretrained
python -m src.eval.extract_embeddings --config configs/config.yaml --model scratch
python -m src.eval.visualize_embeddings --config configs/config.yaml
python -m src.eval.clustering_metrics --config configs/config.yaml
```

On CPU, run everything with a tiny `debug: true` subset (see config) just to
verify the pipeline executes end-to-end before moving to Colab for full runs.
