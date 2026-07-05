"""
Stage 1: SimCLR self-supervised pretraining on unlabeled.csv.

Run:
    python -m src.train.pretrain_simclr --config configs/config.yaml

Saves the pretrained backbone (NOT the projection head — see simclr.py
docstring for why) to outputs/checkpoints/simclr_backbone.pt, plus a loss
curve to outputs/figures/simclr_pretrain_loss.png.
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.utils.config import load_config, get_device
from src.utils.seed import set_seed
from src.data.kitti_dataset import KittiUnlabeledDataset
from src.models.simclr import SimCLRModel
from src.losses.nt_xent import NTXentLoss


def main(config_path: str):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    device = get_device()
    print(f"Using device: {device}")

    scfg = cfg["simclr"]
    debug = cfg["debug"]["enabled"]

    # --- Data ---
    dataset = KittiUnlabeledDataset(
        manifest_csv=cfg["paths"]["unlabeled_manifest"],
        image_size=cfg["image"]["size"],
        debug_max_samples=cfg["debug"]["max_samples"] if debug else None,
    )
    batch_size = min(scfg["batch_size"], len(dataset))
    if batch_size < 2:
        raise RuntimeError(
            f"Need at least 2 samples for contrastive loss, got dataset size {len(dataset)}. "
            "Check unlabeled.csv / debug.max_samples."
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=scfg["num_workers"],
        drop_last=True,  # NT-Xent needs a fixed batch size for the index math
        pin_memory=(device.type == "cuda"),
    )
    print(f"Pretraining on {len(dataset)} unlabeled frames, batch_size={batch_size}, "
          f"{len(loader)} batches/epoch")

    # --- Model / loss / optimizer ---
    model = SimCLRModel(
        pretrained_imagenet=scfg["pretrained_imagenet"],
        hidden_dim=scfg["hidden_dim"],
        projection_dim=scfg["projection_dim"],
    ).to(device)
    criterion = NTXentLoss(temperature=scfg["temperature"])
    optimizer = torch.optim.Adam(model.parameters(), lr=scfg["lr"], weight_decay=scfg["weight_decay"])

    epochs = cfg["debug"]["epochs"] if debug else scfg["epochs"]

    # --- Train ---
    losses = []
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")
        for view1, view2 in pbar:
            view1, view2 = view1.to(device), view2.to(device)

            _, z1 = model(view1)
            _, z2 = model(view2)
            loss = criterion(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch}/{epochs}  avg NT-Xent loss: {avg_loss:.4f}")

    # --- Save backbone only ---
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    backbone_path = ckpt_dir / "simclr_backbone.pt"
    torch.save(model.backbone.state_dict(), backbone_path)
    print(f"Saved pretrained backbone -> {backbone_path}")

    # --- Loss curve ---
    fig_dir = Path(cfg["paths"]["figure_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, epochs + 1), losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("NT-Xent loss")
    plt.title("SimCLR pretraining loss")
    plt.tight_layout()
    fig_path = fig_dir / "simclr_pretrain_loss.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Saved loss curve -> {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
