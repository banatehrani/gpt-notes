# gpt-notes

This repository implements a **character-level GPT language model** following  
Andrej Karpathy’s *“Let’s build GPT”* video lecture, with a clean, modular, and
professional PyTorch codebase.

The intent is **learning-first**, while keeping the structure and code quality
close to what you would expect in a real ML research or engineering project.

---

## What is implemented

- Character-level tokenizer and vocabulary construction  
- Efficient batching for autoregressive language modeling  
- Bigram language model baseline  
- Causal self-attention from scratch  
- Multi-head self-attention  
- Feed-forward networks  
- Transformer blocks (pre-layer normalization)  
- Full GPT language model  
- Training and sampling on the Tiny Shakespeare dataset  

---

## Repository structure

```text
src/gpt_notes/
  data.py        # tokenizer, dataset loading, batching
  attention.py   # attention heads, multi-head attention, transformer blocks
  model.py       # GPT language model
  config.py      # model and training configuration dataclasses
  utils.py       # device selection and reproducibility helpers

experiments/
  get_data.py
  01_data_smoketest.py
  02_train_bigram.py
  03_attention_smoketest.py
  04_block_smoketest.py
  05_train_gpt.py
```

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

---

## Running

```bash
python experiments/get_data.py
python experiments/05_train_gpt.py
```

---

## References

- Andrej Karpathy — *Let’s build GPT*
- https://github.com/karpathy/ng-video-lecture
- https://github.com/karpathy/nanoGPT