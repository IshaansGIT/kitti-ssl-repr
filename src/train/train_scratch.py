"""
Stage 2b: train the SAME architecture from scratch (random init, no SimCLR
pretraining, no ImageNet weights) on labeled.csv. This is the baseline the
fine-tuned model is compared against.

Deliberately mirrors finetune.py's structure as closely as possible — same
data split, same batch size / lr source (scratch config block), same eval
loop — so the only real difference between the two scripts is initialization.

Run:
    python -m src.train.train_scratch --config configs/config.yaml

Saves:
    outputs/checkpoints/scratch_model.pt
    outputs/figures/scratch_curves.png
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.utils.config import load_config, get_device
from src.utils.seed import set_seed
from src.data.labeled_dataset import KittiLabeledDataset
from src.data.split_utils import get_train_val_split
from src.models.classifier import KittiClassifier


def evaluate(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


def main(config_path: str):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    device = get_device()
    print(f"Using device: {device}")

    scfg = cfg["scratch"]
    debug = cfg["debug"]["enabled"]
    image_size = cfg["image"]["size"]

    # --- Data: SAME cached split as finetune.py — critical for a fair comparison ---
    train_df, val_df = get_train_val_split(
        labeled_manifest=cfg["paths"]["labeled_manifest"],
        val_fraction=cfg["data_budget"]["val_fraction"],
        seed=cfg["seed"],
        cache_dir=cfg["paths"]["checkpoint_dir"],
    )
    print(f"Train/val split: {len(train_df)} train, {len(val_df)} val")

    train_ds = KittiLabeledDataset(train_df, image_size=image_size, train=True,
                                    debug_max_samples=cfg["debug"]["max_samples"] if debug else None)
    val_ds = KittiLabeledDataset(val_df, image_size=image_size, train=False,
                                  debug_max_samples=cfg["debug"]["max_samples"] if debug else None)

    batch_size = min(scfg["batch_size"], len(train_ds))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=scfg["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=scfg["num_workers"])

    num_classes = train_ds.num_classes
    print(f"num_classes={num_classes}, class map={train_ds.class_names()}")

    # --- Model: fully random init. No ImageNet, no SimCLR. ---
    model = KittiClassifier(num_classes=num_classes, pretrained_imagenet=False).to(device)
    print("Model initialized from scratch (random weights, no pretraining of any kind)")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=scfg["lr"], weight_decay=scfg["weight_decay"])

    epochs = cfg["debug"]["epochs"] if debug else scfg["epochs"]

    # --- Train ---
    train_losses, val_accuracies = [], []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        val_acc = evaluate(model, val_loader, device)
        train_losses.append(avg_loss)
        val_accuracies.append(val_acc)
        print(f"Epoch {epoch}/{epochs}  train_loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    # --- Save ---
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model_path = ckpt_dir / "scratch_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Saved from-scratch model -> {model_path}")
    print(f"Final val accuracy: {val_accuracies[-1]:.4f}")

    # --- Curves ---
    fig_dir = Path(cfg["paths"]["figure_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(range(1, epochs + 1), train_losses, marker="o", color="orange")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train loss")
    ax1.set_title("From-scratch: training loss")
    ax2.plot(range(1, epochs + 1), val_accuracies, marker="o", color="red")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val accuracy")
    ax2.set_title("From-scratch: validation accuracy")
    ax2.set_ylim(0, 1)
    plt.tight_layout()
    fig_path = fig_dir / "scratch_curves.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Saved curves -> {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
