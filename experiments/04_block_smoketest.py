from __future__ import annotations

import torch

from gpt_notes.attention import TransformerBlock
from gpt_notes.utils import get_device, set_seed


def main() -> None:
    set_seed(1337)
    device = get_device()

    B, T, C = 4, 8, 32
    block_size = 128
    n_head = 4

    x = torch.randn(B, T, C, device=device)
    block = TransformerBlock(n_embd=C, n_head=n_head, block_size=block_size, dropout=0.0).to(device)

    y = block(x)
    print("device:", device)
    print("x:", tuple(x.shape))
    print("y:", tuple(y.shape))
    assert y.shape == x.shape


if __name__ == "__main__":
    main()