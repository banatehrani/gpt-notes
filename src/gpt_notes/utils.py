from __future__ import annotations

import torch


def get_device() -> torch.device:
    # Mac: MPS if available, else CPU
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)