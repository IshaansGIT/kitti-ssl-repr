"""
Extracts backbone features (the 512-d pooled ResNet-18 output, pre-classifier)
for every image in labeled.csv, using either the fine-tuned or from-scratch
model's backbone. Saves embeddings + labels + categories to a .npz file for
visualize_embeddings.py and clustering_metrics.py to consume.

We deliberately extract from the FULL labeled set (train+val combined, 300
images), not just val — t-SNE/PCA are about visualizing representation
structure, not measuring generalization, so there's no leakage concern here
the way there would be for an accuracy number.

Run:
    python -m src.eval.extract_embeddings --config configs/config.yaml --model pretrained
    python -m src.eval.extract_embeddings --config configs/config.yaml --model scratch

Saves:
    outputs/checkpoints/embeddings_pretrained.npz
    outputs/checkpoints/embeddings_scratch.npz
    each containing: embeddings (N, 512), labels (N,), categories (N,) [strings]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.utils.config import load_config, get_device
from src.utils.seed import set_seed
from src.data.labeled_dataset import KittiLabeledDataset
from src.models.classifier import KittiClassifier


def main(config_path: str, model_name: str):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    device = get_device()
    print(f"Using device: {device}")

    image_size = cfg["image"]["size"]
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])

    # --- Load the full labeled manifest (train+val combined) ---
    full_df = pd.read_csv(cfg["paths"]["labeled_manifest"])
    dataset = KittiLabeledDataset(full_df, image_size=image_size, train=False)  # eval transform: deterministic
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    num_classes = dataset.num_classes
    class_names = dataset.class_names()

    # --- Build model and load the requested checkpoint ---
    model = KittiClassifier(num_classes=num_classes, pretrained_imagenet=False).to(device)

    if model_name == "pretrained":
        ckpt_path = ckpt_dir / "finetuned_model.pt"
    elif model_name == "scratch":
        ckpt_path = ckpt_dir / "scratch_model.pt"
    else:
        raise ValueError(f"Unknown --model {model_name!r}, expected 'pretrained' or 'scratch'")

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path}. Run finetune.py / train_scratch.py first."
        )
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded {model_name} model from {ckpt_path}")

    # --- Extract backbone features (pre-classifier) for every image ---
    model.eval()
    all_embeddings, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            features = model.backbone(images)  # (batch, 512), classifier head not applied
            all_embeddings.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    categories = np.array([class_names[int(l)] for l in labels])

    print(f"Extracted embeddings: {embeddings.shape}")

    out_path = ckpt_dir / f"embeddings_{model_name}.npz"
    np.savez(out_path, embeddings=embeddings, labels=labels, categories=categories)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--model", type=str, required=True, choices=["pretrained", "scratch"])
    args = parser.parse_args()
    main(args.config, args.model)
