from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttentionHead(nn.Module):
    """
    Single self-attention head with a causal (lower-triangular) mask.
    Input:  x (B, T, C)
    Output: (B, T, head_size)
    """

    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # Register mask as a buffer so it moves with device and is saved in state_dict
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)
        self.head_size = head_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        k = self.key(x)    # (B, T, hs)
        q = self.query(x)  # (B, T, hs)

        # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)

        # causal mask: prevent attending to future positions
        mask = self.tril[:T, :T]  # (T, T)
        wei = wei.masked_fill(mask == 0, float("-inf"))

        wei = F.softmax(wei, dim=-1)          # (B, T, T)
        wei = self.dropout(wei)

        v = self.value(x)                      # (B, T, hs)
        out = wei @ v                          # (B, T, hs)
        return out
    
class MultiHeadAttention(nn.Module):
    """
    Multiple causal self-attention heads in parallel + projection back to n_embd.
    Input:  x (B, T, C=n_embd)
    Output: (B, T, C=n_embd)
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        head_size = n_embd // n_head

        self.heads = nn.ModuleList(
            [
                CausalSelfAttentionHead(
                    n_embd=n_embd,
                    head_size=head_size,
                    block_size=block_size,
                    dropout=dropout,
                )
                for _ in range(n_head)
            ]
        )
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, n_embd)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    """
    Simple MLP used inside Transformer blocks.
    """

    def __init__(self, n_embd: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Pre-LN Transformer block (more stable / modern than post-LN).
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.sa = MultiHeadAttention(n_embd=n_embd, n_head=n_head, block_size=block_size, dropout=dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd=n_embd, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x