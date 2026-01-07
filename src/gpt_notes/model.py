from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from gpt_notes.attention import TransformerBlock
from gpt_notes.config import GPTConfig


class GPTLanguageModel(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    n_embd=cfg.n_embd,
                    n_head=cfg.n_head,
                    block_size=cfg.block_size,
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size)

        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        assert T <= self.cfg.block_size, "Sequence length exceeds block_size"

        tok = self.token_emb(idx)  # (B, T, C)
        pos = self.pos_emb(torch.arange(T, device=idx.device))  # (T, C)
        x = tok + pos  # broadcast (B, T, C)
        x = self.dropout(x)

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]  # crop context window
            logits, _ = self(idx_cond)
            logits_last = logits[:, -1, :]  # (B, vocab)
            probs = F.softmax(logits_last, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_idx], dim=1)
        return idx