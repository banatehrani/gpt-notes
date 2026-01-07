from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 64
    block_size: int = 128
    max_iters: int = 5000
    eval_interval: int = 250
    eval_iters: int = 50
    learning_rate: float = 3e-4
    seed: int = 1337


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    block_size: int
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.1