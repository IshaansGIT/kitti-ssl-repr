"""
Unlabeled dataset for SimCLR pretraining.

Each __getitem__ returns TWO independently augmented views of the same
underlying frame — that pair is what the NT-Xent loss pulls together while
pushing apart views from other frames in the batch.
"""
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def simclr_augmentation(image_size: int) -> transforms.Compose:
    """
    Standard SimCLR augmentation recipe, adapted for driving-scene images:
    random resized crop, horizontal flip, color jitter, occasional grayscale
    and blur. We keep crop scale fairly mild (0.5-1.0) — KITTI frames are
    wide and scene-defining structure (road vs building layout) can vanish
    under aggressive cropping, which would make the pretext task too easy
    but for the wrong reasons (or too hard/noisy) at our very small ~500
    image scale.
    """
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=9)], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class KittiUnlabeledDataset(Dataset):
    """Reads paths from a manifest CSV (column: `path`), returns (view1, view2)."""

    def __init__(self, manifest_csv: str, image_size: int = 224, debug_max_samples: int | None = None):
        self.df = pd.read_csv(manifest_csv)
        if debug_max_samples is not None:
            self.df = self.df.iloc[:debug_max_samples].reset_index(drop=True)
        self.transform = simclr_augmentation(image_size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        path = self.df.iloc[idx]["path"]
        image = Image.open(path).convert("RGB")
        view1 = self.transform(image)
        view2 = self.transform(image)
        return view1, view2
