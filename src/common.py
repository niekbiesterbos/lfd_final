import os
from pathlib import Path
import random
import numpy as np
import torch


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist; does nothing if it already exists."""
    Path(path).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    """Set all relevant RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Enforce deterministic behavior where feasible; may have performance impact.
    torch.use_deterministic_algorithms(True, warn_only=True)
