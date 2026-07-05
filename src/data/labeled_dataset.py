"""
Labeled dataset for the small-data fine-tune vs from-scratch comparison.

Reads the `labeled.csv` manifest (columns: path, label, category) produced by
prepare_data.py. Supports a `train` flag to switch between light train-time
augmentation and deterministic eval-time preprocessing.
"""
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def train_transform(image_size: int) -> transforms.Compose:
    # Lighter than SimCLR's augmentation — this is supervised training on a
    # tiny labeled set, not contrastive pretraining, so we want mild
    # regularization, not aggressive view generation.
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class KittiLabeledDataset(Dataset):

    def __init__(
        self,
        manifest_csv_or_df,
        image_size: int = 224,
        train: bool = True,
        debug_max_samples: int | None = None,
    ):
        """
        manifest_csv_or_df: either a path to a manifest CSV, or an already-
        loaded DataFrame (e.g. one half of a train/val split from
        split_utils.get_train_val_split).
        """
        if isinstance(manifest_csv_or_df, pd.DataFrame):
            self.df = manifest_csv_or_df.reset_index(drop=True)
        else:
            self.df = pd.read_csv(manifest_csv_or_df)
        if debug_max_samples is not None:
            self.df = self.df.iloc[:debug_max_samples].reset_index(drop=True)
        self.transform = train_transform(image_size) if train else eval_transform(image_size)
        self.num_classes = self.df["label"].nunique()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        image = self.transform(image)
        return image, int(row["label"])

    def class_names(self) -> dict[int, str]:
        return dict(zip(self.df["label"], self.df["category"]))