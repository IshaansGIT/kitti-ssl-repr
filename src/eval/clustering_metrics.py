"""
Quantifies clustering quality on the raw (512-d, pre-t-SNE/PCA) embeddings
using ground-truth category labels as cluster assignments. This complements
the visual t-SNE/PCA inspection with numbers.

  Silhouette score: range [-1, 1], higher is better (well-separated clusters).
  Davies-Bouldin index: >= 0, LOWER is better (unlike most metrics here) --
      measures average similarity between each cluster and its most similar
      other cluster.

Run (after extract_embeddings.py has been run for both models):
    python -m src.eval.clustering_metrics --config configs/config.yaml
"""
import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score

from src.utils.config import load_config


def main(config_path: str):
    cfg = load_config(config_path)
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])

    print(f"{'Model':<12} {'Silhouette (higher better)':<28} {'Davies-Bouldin (lower better)'}")
    for model_name in ["pretrained", "scratch"]:
        path = ckpt_dir / f"embeddings_{model_name}.npz"
        if not path.exists():
            print(f"{model_name:<12} [missing embeddings -- run extract_embeddings.py first]")
            continue
        data = np.load(path, allow_pickle=True)
        embeddings, labels = data["embeddings"], data["labels"]

        sil = silhouette_score(embeddings, labels)
        db = davies_bouldin_score(embeddings, labels)
        print(f"{model_name:<12} {sil:<28.4f} {db:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
