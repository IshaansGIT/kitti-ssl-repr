"""
Scans the KITTI Raw sequences listed in configs/config.yaml and produces two
manifest CSVs:

  data/manifests/unlabeled.csv   columns: [path]
      Frames for self-supervised pretraining. No labels used.

  data/manifests/labeled.csv     columns: [path, label, category]
      A small, roughly category-balanced subset for the fine-tune vs
      from-scratch comparison. `label` is an integer class id, `category`
      is the human-readable KITTI category name.

Run:
    python -m src.data.prepare_data --config configs/config.yaml
"""
import argparse
import random
from pathlib import Path

import pandas as pd

from src.utils.config import load_config
from src.utils.seed import set_seed


def find_frames(raw_root: Path, seq: dict, stride: int) -> list[Path]:
    """Return every `stride`-th frame path for one sequence, sorted by filename."""
    seq_dir = raw_root / seq["date"] / seq["drive"] / seq["camera_dir"] / "data"
    if not seq_dir.exists():
        print(f"  [WARN] missing sequence dir, skipping: {seq_dir}")
        return []
    frames = sorted(seq_dir.glob("*.png"))
    if not frames:
        frames = sorted(seq_dir.glob("*.jpg"))
    return frames[::stride]


def main(config_path: str):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])

    raw_root = Path(cfg["paths"]["raw_root"])
    budget = cfg["data_budget"]
    stride = budget["frame_stride"]

    # 1. Collect all available frames, tagged with their category.
    per_sequence_frames = []  # list of (category, [paths])
    for seq in cfg["sequences"]:
        frames = find_frames(raw_root, seq, stride)
        print(f"  {seq['drive']} [{seq['category']}]: {len(frames)} frames after stride={stride}")
        if frames:
            per_sequence_frames.append((seq["category"], frames))

    if not per_sequence_frames:
        raise RuntimeError(
            "No frames found. Check configs/config.yaml `sequences:` and "
            "confirm data/raw/<date>/<drive>/<camera_dir>/data/*.png exists."
        )

    all_frames = [p for _, frames in per_sequence_frames for p in frames]
    random.shuffle(all_frames)

    # 2. Unlabeled manifest: random sample across everything, capped by budget.
    n_unlabeled = min(len(all_frames), budget["max_unlabeled_frames"])
    unlabeled_paths = all_frames[:n_unlabeled]
    unlabeled_df = pd.DataFrame({"path": [str(p) for p in unlabeled_paths]})

    # 3. Labeled manifest: sample roughly evenly per category, capped by budget.
    by_category: dict[str, list[Path]] = {}
    for category, frames in per_sequence_frames:
        by_category.setdefault(category, []).extend(frames)
    for frames in by_category.values():
        random.shuffle(frames)

    categories = sorted(by_category.keys())
    label_map = {cat: i for i, cat in enumerate(categories)}
    per_cat_cap = budget["labeled_frames_per_category"]

    labeled_rows = []
    for category in categories:
        take = by_category[category][:per_cat_cap]
        for p in take:
            labeled_rows.append({"path": str(p), "label": label_map[category], "category": category})

    labeled_df = pd.DataFrame(labeled_rows)
    if len(labeled_df) > budget["max_labeled_frames"]:
        labeled_df = (
            labeled_df.groupby("category", group_keys=False)
            .apply(lambda g: g.sample(frac=1, random_state=cfg["seed"]))
            .reset_index(drop=True)
            .iloc[: budget["max_labeled_frames"]]
        )

    # 4. Write out.
    out_unlabeled = Path(cfg["paths"]["unlabeled_manifest"])
    out_labeled = Path(cfg["paths"]["labeled_manifest"])
    out_unlabeled.parent.mkdir(parents=True, exist_ok=True)
    unlabeled_df.to_csv(out_unlabeled, index=False)
    labeled_df.to_csv(out_labeled, index=False)

    print(f"\nWrote {len(unlabeled_df)} unlabeled frames -> {out_unlabeled}")
    print(f"Wrote {len(labeled_df)} labeled frames -> {out_labeled}")
    print("Label map:", label_map)
    print(labeled_df["category"].value_counts())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
