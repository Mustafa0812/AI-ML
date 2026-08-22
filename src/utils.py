"""
utils.py
Small shared helpers: reproducibility and device selection.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Fix all relevant RNG seeds for reproducibility (report requirement)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Pick GPU if available, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")
