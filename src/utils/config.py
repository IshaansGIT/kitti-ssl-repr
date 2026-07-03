"""Small YAML config loader shared by every script.

Usage:
    from src.utils.config import load_config, get_device
    cfg = load_config("configs/config.yaml")
    cfg["simclr"]["lr"]        # plain dict access
    device = get_device()
"""
from pathlib import Path
import yaml


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    # Resolve paths relative to the repo root (parent of `configs/`), so the
    # scripts work the same whether invoked from repo root or elsewhere.
    repo_root = Path(path).resolve().parent.parent
    for key, val in cfg["paths"].items():
        cfg["paths"][key] = str(repo_root / val)

    return cfg


def get_device():
    # Imported lazily so scripts that only need config parsing (e.g. data prep)
    # don't require torch to be installed at all.
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon, still much faster than CPU
        return torch.device("mps")
    return torch.device("cpu")
