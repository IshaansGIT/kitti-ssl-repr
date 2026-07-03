"""Reproducibility helper. Call set_seed() once at the top of every entrypoint."""
import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        # Optional: only data-prep scripts run without torch installed.
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
    except ImportError:
        pass
