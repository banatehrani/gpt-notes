from __future__ import annotations

import time

import torch

from gpt_notes.bigram import BigramConfig, BigramLanguageModel
from gpt_notes.data import CharTokenizer, get_batch, load_text, make_splits


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def estimate_loss(
    model: BigramLanguageModel,
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
    # Reproducibility
    torch.manual_seed(1337)

    device = get_device()
    print("device:", device)

    text = load_text("data/tinyshakespeare.txt")
    tok = CharTokenizer.from_text(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)

    train_ids, val_ids = make_splits(ids, train_frac=0.9)

    # Hyperparams (small baseline)
    batch_size = 64
    block_size = 128
    max_iters = 2000
    eval_interval = 200
    lr = 1e-2

    cfg = BigramConfig(vocab_size=tok.vocab_size)
    model = BigramLanguageModel(cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    t0 = time.time()
    for step in range(1, max_iters + 1):
        if step % eval_interval == 0 or step == 1:
            losses = estimate_loss(
                model, train_ids, val_ids, block_size, batch_size, device
            )
            dt = time.time() - t0
            print(
                f"step {step:5d} | train {losses['train']:.4f} | "
                f"val {losses['val']:.4f} | elapsed {dt:.1f}s"
            )

        xb, yb = get_batch(train_ids, block_size, batch_size, device)
        _, loss = model(xb, yb)
        assert loss is not None

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # sample
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = model.generate(context, max_new_tokens=500)[0].tolist()
    print("\n--- sample ---\n")
    print(tok.decode(out))


if __name__ == "__main__":
    main()