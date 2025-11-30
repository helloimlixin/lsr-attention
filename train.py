"""Training script for GPT language model with LSR attention."""

import math
import argparse
from dataclasses import dataclass
import time
import random
import numpy as np

import torch

from data import load_wikitext, make_batch
from model import GPTLM


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class Config:
    """Training configuration."""
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 8
    d_ff: int = 2048
    max_seq_len: int = 256
    batch_size: int = 32
    lr: float = 3e-4
    steps: int = 5000
    eval_interval: int = 500
    eval_batches: int = 50
    attn_type: str = "lsr"
    lsr_rank: int = 32
    dataset_name: str = "wikitext-2-raw-v1"
    device: str = "cuda"
    seed: int = 0


def evaluate(model, tokens, cfg: Config):
    """Evaluate model on validation set."""
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(cfg.eval_batches):
            x, y = make_batch(tokens, cfg.batch_size, cfg.max_seq_len, cfg.device)
            _, loss = model(x, y)
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    cfg = Config(
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        max_seq_len=args.seq_len,
        batch_size=args.batch_size,
        lr=args.lr,
        steps=args.steps,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        attn_type=args.attn_type,
        lsr_rank=args.lsr_rank,
        dataset_name=args.dataset,
        device=device,
        seed=args.seed,
    )

    set_seed(cfg.seed)

    print("Config:", cfg)
    print("Using device:", cfg.device)
    print("Loading dataset + tokenizer:", cfg.dataset_name)
    
    tokenizer, train_ids, val_ids = load_wikitext(cfg.dataset_name)
    vocab_size = tokenizer.vocab_size
    train_ids = train_ids.to(cfg.device)
    val_ids = val_ids.to(cfg.device)

    print(f"Vocab size: {vocab_size}, train tokens: {train_ids.numel()}, val tokens: {val_ids.numel()}")

    model = GPTLM(
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        d_ff=cfg.d_ff,
        max_seq_len=cfg.max_seq_len,
        attn_type=cfg.attn_type,
        lsr_rank=cfg.lsr_rank,
    ).to(cfg.device)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    tokens_per_step = cfg.batch_size * cfg.max_seq_len
    print("Tokens per step:", tokens_per_step)

    print("Starting training...")
    t0 = time.time()
    
    for step in range(1, cfg.steps + 1):
        x, y = make_batch(train_ids, cfg.batch_size, cfg.max_seq_len, cfg.device)
        _, loss = model(x, y)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        if step % 50 == 0:
            elapsed = time.time() - t0
            toks = step * tokens_per_step
            toks_per_sec = toks / max(elapsed, 1e-8)
            print(f"step {step}/{cfg.steps} "
                  f"| train loss {loss.item():.4f} "
                  f"| tok/s {toks_per_sec:.1f}")

        if step % cfg.eval_interval == 0:
            val_loss = evaluate(model, val_ids, cfg)
            ppl = math.exp(val_loss)
            print(f"[EVAL] step {step} | val loss {val_loss:.4f} | ppl {ppl:.2f}")

    t1 = time.time()
    total_time = t1 - t0
    total_tokens = cfg.steps * tokens_per_step
    toks_per_sec = total_tokens / max(total_time, 1e-8)
    
    print("Done.")
    print(f"Total train time: {total_time:.1f} s "
          f"| total tokens: {total_tokens} "
          f"| avg tok/s: {toks_per_sec:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GPT with LSR attention")
    
    # Attention
    parser.add_argument("--attn_type", type=str, default="lsr",
                        choices=["dot", "lsr", "lsr_triton"], help="Attention type")
    parser.add_argument("--lsr_rank", type=int, default=32,
                        help="Rank for LSR attention")

    # Model architecture
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--seq_len", type=int, default=1024)
    
    # Training
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--eval_batches", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    
    # Data
    parser.add_argument("--dataset", type=str, default="wikitext-2-raw-v1",
                        choices=["wikitext-2-raw-v1", "wikitext-103-raw-v1"])
    
    # Misc
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()
    main(args)
