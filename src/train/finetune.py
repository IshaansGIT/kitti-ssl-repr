"""
Stage 2a: fine-tune the SimCLR-pretrained backbone on labeled.csv.

Loads outputs/checkpoints/simclr_backbone.pt (produced by pretrain_simclr.py)
into a fresh KittiClassifier, then trains on the labeled train split and
evaluates on the held-out val split every epoch.

Run:
    python -m src.train.finetune --config configs/config.yaml

Saves:
    outputs/checkpoints/finetuned_model.pt
    outputs/figures/finetune_curves.png       (train loss + val accuracy)
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

    fcfg = cfg["finetune"]
    debug = cfg["debug"]["enabled"]
    image_size = cfg["image"]["size"]

    # --- Data: shared, cached train/val split (same split train_scratch.py uses) ---
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

    batch_size = min(fcfg["batch_size"], len(train_ds))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=fcfg["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=fcfg["num_workers"])

    num_classes = train_ds.num_classes
    print(f"num_classes={num_classes}, class map={train_ds.class_names()}")

    # --- Model: build with random-init backbone, then load SimCLR weights ---
    model = KittiClassifier(num_classes=num_classes, pretrained_imagenet=False).to(device)
    backbone_ckpt = Path(cfg["paths"]["checkpoint_dir"]) / "simclr_backbone.pt"
    if not backbone_ckpt.exists():
        raise FileNotFoundError(
            f"No pretrained backbone found at {backbone_ckpt}. "
            "Run pretrain_simclr.py first."
        )
    state_dict = torch.load(backbone_ckpt, map_location=device)
    model.load_backbone_state_dict(state_dict)
    print(f"Loaded pretrained backbone from {backbone_ckpt}")

    if fcfg["freeze_backbone"]:
        model.freeze_backbone()
        print("Backbone frozen: linear-probe mode (only classifier head trains)")
    else:
        print("Backbone unfrozen: full fine-tuning")

    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=fcfg["lr"], weight_decay=fcfg["weight_decay"])

    epochs = cfg["debug"]["epochs"] if debug else fcfg["epochs"]

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
    model_path = ckpt_dir / "finetuned_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Saved fine-tuned model -> {model_path}")
    print(f"Final val accuracy: {val_accuracies[-1]:.4f}")

    # --- Curves ---
    fig_dir = Path(cfg["paths"]["figure_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(range(1, epochs + 1), train_losses, marker="o")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train loss")
    ax1.set_title("Fine-tune: training loss")
    ax2.plot(range(1, epochs + 1), val_accuracies, marker="o", color="green")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val accuracy")
    ax2.set_title("Fine-tune: validation accuracy")
    ax2.set_ylim(0, 1)
    plt.tight_layout()
    fig_path = fig_dir / "finetune_curves.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Saved curves -> {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
