from __future__ import annotations

import torch

from gpt_notes.data import CharTokenizer, get_batch, load_text, make_splits


def main() -> None:
    text = load_text("data/tinyshakespeare.txt")
    tok = CharTokenizer.from_text(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)

    train_ids, val_ids = make_splits(ids, train_frac=0.9)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    x, y = get_batch(train_ids, block_size=64, batch_size=4, device=device)

    print("vocab_size:", tok.vocab_size)
    print("device:", device)
    print("x shape:", tuple(x.shape), "y shape:", tuple(y.shape))
    print("decoded x[0][:100]:")
    print(tok.decode(x[0].tolist())[:100])


if __name__ == "__main__":
    main()