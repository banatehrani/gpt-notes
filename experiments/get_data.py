from __future__ import annotations

import pathlib
import urllib.request

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
OUT_PATH = pathlib.Path("data") / "tinyshakespeare.txt"


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUT_PATH.exists() and OUT_PATH.stat().st_size > 0:
        print(f"Already exists: {OUT_PATH} ({OUT_PATH.stat().st_size:,} bytes)")
        return

    print(f"Downloading {DATA_URL} -> {OUT_PATH}")
    with urllib.request.urlopen(DATA_URL) as r:
        OUT_PATH.write_bytes(r.read())

    print(f"Done: {OUT_PATH} ({OUT_PATH.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()