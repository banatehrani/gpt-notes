from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class BigramConfig:
    vocab_size: int


class BigramLanguageModel(nn.Module):
    """
    Predict next token using only the current token.
    logits: (B, T, vocab)
    """

    def __init__(self, cfg: BigramConfig) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.vocab_size)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits = self.token_embedding_table(idx)  # (B, T, C)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_2d = logits.view(B * T, C)
            targets_1d = targets.view(B * T)
            loss = F.cross_entropy(logits_2d, targets_1d)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits_last = logits[:, -1, :]           # (B, C)
            probs = F.softmax(logits_last, dim=-1)   # (B, C)
            next_idx = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, next_idx], dim=1)
        return idx