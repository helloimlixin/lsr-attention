"""Data loading and batching utilities."""

import torch
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from pathlib import Path
import urllib.request


def load_wikitext(dataset_name="wikitext-2-raw-v1", tokenizer_name="gpt2"):
    """
    Load WikiText dataset (2 or 103) and tokenize with GPT-2 tokenizer.
    
    Args:
        dataset_name: "wikitext-2-raw-v1" or "wikitext-103-raw-v1"
        tokenizer_name: HuggingFace tokenizer name
    
    Returns:
        tokenizer, train_tokens, val_tokens (1D LongTensors)
    """
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 1_000_000_000  # avoid HF warnings

    ds = load_dataset("wikitext", dataset_name)

    train_text = "\n\n".join(ds["train"]["text"])
    val_text = "\n\n".join(ds["validation"]["text"])

    train_ids = tokenizer(train_text, return_tensors="pt").input_ids[0]
    val_ids = tokenizer(val_text, return_tensors="pt").input_ids[0]

    return tokenizer, train_ids, val_ids


def make_batch(tokens, batch_size, seq_len, device):
    """
    Sample random contiguous chunks from a 1D token tensor.
    
    Args:
        tokens: 1D tensor of token ids
        batch_size: number of sequences per batch
        seq_len: length of each sequence
        device: target device for output tensors
    
    Returns:
        x, y: input and target tensors of shape (B, T)
    """
    max_start = tokens.size(0) - (seq_len + 1)
    idx = torch.randint(0, max_start, (batch_size,), device=tokens.device)

    offsets = torch.arange(seq_len, device=tokens.device).unsqueeze(0)
    starts = idx.unsqueeze(1)
    positions = starts + offsets

    x = tokens[positions]
    y = tokens[positions + 1]
    return x.to(device), y.to(device)


def load_shakespeare(tokenizer_name="gpt2", text_path: str | None = None):
    """Load tiny Shakespeare (Karpathy-style) and tokenize with GPT-2 tokenizer.

    Args:
        tokenizer_name: HF tokenizer name.
        text_path: optional path to a local text file (e.g., inputs.txt). If None, will
            download the standard tinyshakespeare input.txt to cache.
    """
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 1_000_000_000

    if text_path is not None:
        cache_path = Path(text_path)
        if cache_path.exists() and cache_path.is_dir():
            raise ValueError(f"Provided text_path is a directory, not a file: {cache_path}")
        if not cache_path.exists():
            # Download tinyshakespeare to the requested location
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, cache_path)
    else:
        cache_dir = Path.home() / ".cache" / "lsr_attention"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / "tinyshakespeare.txt"
        if not cache_path.exists():
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            urllib.request.urlretrieve(url, cache_path)

    text = cache_path.read_text(encoding="utf-8")
    split = int(0.9 * len(text))
    train_text = text[:split]
    val_text = text[split:]

    train_ids = tokenizer(train_text, return_tensors="pt").input_ids[0]
    val_ids = tokenizer(val_text, return_tensors="pt").input_ids[0]

    return tokenizer, train_ids, val_ids

