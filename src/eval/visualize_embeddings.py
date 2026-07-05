"""
Projects the extracted embeddings (from extract_embeddings.py) down to 2D
with both PCA and t-SNE, and plots pretrained vs from-scratch side by side,
colored by category. This is the core visualization the assignment asks for:
"show clustering quality visually."

Run (after extract_embeddings.py has been run for both models):
    python -m src.eval.visualize_embeddings --config configs/config.yaml

Saves:
    outputs/figures/embeddings_comparison.png   (2x2 grid: rows=model, cols=PCA/t-SNE)
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.utils.config import load_config
from src.utils.seed import set_seed


CATEGORY_COLORS = {"City": "#e74c3c", "Residential": "#2ecc71", "Road": "#3498db",
                    "Campus": "#9b59b6", "Person": "#f39c12"}


def load_embeddings(ckpt_dir: Path, model_name: str):
    path = ckpt_dir / f"embeddings_{model_name}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"No embeddings file at {path}. Run: "
            f"python -m src.eval.extract_embeddings --config configs/config.yaml --model {model_name}"
        )
    data = np.load(path, allow_pickle=True)
    return data["embeddings"], data["labels"], data["categories"]


def plot_projection(ax, points_2d: np.ndarray, categories: np.ndarray, title: str):
    for category in sorted(set(categories)):
        mask = categories == category
        color = CATEGORY_COLORS.get(category, "gray")
        ax.scatter(points_2d[mask, 0], points_2d[mask, 1], label=category,
                   color=color, alpha=0.7, s=25, edgecolors="none")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=8, loc="best")


def main(config_path: str):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    fig_dir = Path(cfg["paths"]["figure_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for row, model_name in enumerate(["pretrained", "scratch"]):
        embeddings, labels, categories = load_embeddings(ckpt_dir, model_name)
        print(f"{model_name}: {embeddings.shape[0]} embeddings, dim={embeddings.shape[1]}")

        # PCA: linear projection preserving global variance directions
        pca = PCA(n_components=2, random_state=cfg["seed"])
        pca_2d = pca.fit_transform(embeddings)
        explained = pca.explained_variance_ratio_.sum()
        plot_projection(
            axes[row, 0], pca_2d, categories,
            f"{model_name.capitalize()} — PCA ({explained:.1%} variance explained)",
        )

        # t-SNE: nonlinear, preserves local neighborhood structure -- usually
        # gives visually tighter/cleaner clusters than PCA for this kind of
        # data, at the cost of not preserving global distances meaningfully.
        n_samples = embeddings.shape[0]
        perplexity = min(30, max(5, n_samples // 10))  # sane default for small N
        tsne = TSNE(n_components=2, random_state=cfg["seed"], perplexity=perplexity, init="pca")
        tsne_2d = tsne.fit_transform(embeddings)
        plot_projection(
            axes[row, 1], tsne_2d, categories,
            f"{model_name.capitalize()} — t-SNE (perplexity={perplexity})",
        )

    plt.suptitle("Embedding structure: pretrained (SimCLR + fine-tune) vs from-scratch", fontsize=13)
    plt.tight_layout()
    out_path = fig_dir / "embeddings_comparison.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
