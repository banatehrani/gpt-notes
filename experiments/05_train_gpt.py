from __future__ import annotations

import time

import torch

from gpt_notes.config import GPTConfig, TrainConfig
from gpt_notes.data import CharTokenizer, get_batch, load_text, make_splits
from gpt_notes.model import GPTLanguageModel
from gpt_notes.utils import get_device, set_seed


@torch.no_grad()
def estimate_loss(
    model: GPTLanguageModel,
    train_ids: torch.Tensor,
    val_ids: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device,
    eval_iters: int = 50,
) -> dict[str, float]:
    model.eval()
    out: dict[str, float] = {}
    for name, data in [("train", train_ids), ("val", val_ids)]:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            assert loss is not None
            losses[k] = loss
        out[name] = float(losses.mean().item())
    model.train()
    return out


def main() -> None:
    train_cfg = TrainConfig(
        batch_size=64,
        block_size=128,
        max_iters=3000,
        eval_interval=300,
        eval_iters=50,
        learning_rate=3e-4,
        seed=1337,
    )

    set_seed(train_cfg.seed)
    device = get_device()
    print("device:", device)

    text = load_text("data/tinyshakespeare.txt")
    tok = CharTokenizer.from_text(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)

    train_ids, val_ids = make_splits(ids, train_frac=0.9)

    model_cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        block_size=train_cfg.block_size,
        n_embd=128,
        n_head=4,
        n_layer=4,
        dropout=0.1,
    )

    model = GPTLanguageModel(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.learning_rate)

    t0 = time.time()
    for step in range(1, train_cfg.max_iters + 1):
        if step % train_cfg.eval_interval == 0 or step == 1:
            losses = estimate_loss(
                model,
                train_ids,
                val_ids,
                train_cfg.block_size,
                train_cfg.batch_size,
                device,
                eval_iters=train_cfg.eval_iters,
            )
            dt = time.time() - t0
            print(
                f"step {step:5d} | train {losses['train']:.4f} | "
                f"val {losses['val']:.4f} | elapsed {dt:.1f}s"
            )

        xb, yb = get_batch(train_ids, train_cfg.block_size, train_cfg.batch_size, device)
        _, loss = model(xb, yb)
        assert loss is not None

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # sample
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = model.generate(context, max_new_tokens=600)[0].tolist()
    print("\n--- sample ---\n")
    print(tok.decode(out))


if __name__ == "__main__":
    main()