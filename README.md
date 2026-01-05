# gpt-notes

Code + notes following Andrej Karpathy's GPT video lecture, with small modernizations:
- clean project layout
- reproducible training runs
- MPS (Apple Silicon) support when available

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"