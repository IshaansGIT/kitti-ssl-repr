"""
Splits labeled.csv into train/val once, deterministically, and caches the
result. Both finetune.py and train_scratch.py call this — they MUST see the
exact same split, otherwise the pretrained-vs-scratch comparison is comparing
different data, not different init.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def get_train_val_split(
    labeled_manifest: str,
    val_fraction: float,
    seed: int,
    cache_dir: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (train_df, val_df). Splits are stratified by label so each class
    is proportionally represented in both sets (important since our labeled
    set is small — a non-stratified split could easily leave a class
    under-represented in val by chance).

    The split is cached to disk on first call so that re-running finetune.py
    and train_scratch.py in separate processes/sessions still uses the
    identical split, not just within a single run.
    """
    cache_path = Path(cache_dir) / "train_val_split_cache.csv"

    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        train_df = cached[cached["split"] == "train"].drop(columns=["split"]).reset_index(drop=True)
        val_df = cached[cached["split"] == "val"].drop(columns=["split"]).reset_index(drop=True)
        return train_df, val_df

    df = pd.read_csv(labeled_manifest)
    train_df, val_df = train_test_split(
        df,
        test_size=val_fraction,
        random_state=seed,
        stratify=df["label"],
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    # Cache with a split column so both scripts (and future re-runs) agree.
    cache_df = pd.concat(
        [train_df.assign(split="train"), val_df.assign(split="val")], ignore_index=True
    )
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache_df.to_csv(cache_path, index=False)

    return train_df, val_df
