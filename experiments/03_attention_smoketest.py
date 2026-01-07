from __future__ import annotations

import torch

from gpt_notes.attention import CausalSelfAttentionHead
from gpt_notes.utils import get_device, set_seed


def main() -> None:
    set_seed(1337)
    device = get_device()

    B, T, C = 4, 8, 32
    head_size = 16
    block_size = 128

    x = torch.randn(B, T, C, device=device)
    head = CausalSelfAttentionHead(n_embd=C, head_size=head_size, block_size=block_size, dropout=0.0).to(
        device
    )

    y = head(x)
    print("device:", device)
    print("x:", tuple(x.shape))
    print("y:", tuple(y.shape))
    assert y.shape == (B, T, head_size)


if __name__ == "__main__":
    main()