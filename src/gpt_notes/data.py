from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class CharTokenizer:
    stoi: dict[str, int]  # string-to-int
    itos: dict[int, str]  # int-to-string

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for ch, i in stoi.items()}
        return cls(stoi=stoi, itos=itos)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


def load_text(path: str | Path) -> str:
    p = Path(path)
    return p.read_text(encoding="utf-8")


def make_splits(
    data_ids: torch.Tensor, train_frac: float = 0.9
) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(train_frac * len(data_ids))
    return data_ids[:n], data_ids[n:]


def get_batch(
    split_data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      x: (B, T) token ids
      y: (B, T) next-token ids
    """
    n = len(split_data)
    ix = torch.randint(0, n - block_size - 1, (batch_size,))
    x = torch.stack([split_data[i : i + block_size] for i in ix])
    y = torch.stack([split_data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)