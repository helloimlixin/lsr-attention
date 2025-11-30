"""GPT language model implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import MultiHeadSelfAttention


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class DecoderBlock(nn.Module):
    """Transformer decoder block with pre-norm architecture."""
    
    def __init__(self, d_model, num_heads, d_ff, attn_type, lsr_rank, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(
            d_model, num_heads,
            attn_type=attn_type,
            lsr_rank=lsr_rank
        )
        self.ff = FeedForward(d_model, d_ff, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPTLM(nn.Module):
    """
    GPT-style language model.
    
    Args:
        vocab_size: vocabulary size
        d_model: model dimension
        num_heads: number of attention heads
        num_layers: number of transformer layers
        d_ff: feed-forward hidden dimension
        max_seq_len: maximum sequence length
        attn_type: "dot" or "lsr"
        lsr_rank: rank for LSR attention
        dropout: dropout probability
    """
    
    def __init__(
        self,
        vocab_size,
        d_model=512,
        num_heads=8,
        num_layers=8,
        d_ff=2048,
        max_seq_len=256,
        attn_type="dot",
        lsr_rank=32,
        dropout=0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList([
            DecoderBlock(
                d_model, num_heads, d_ff,
                attn_type=attn_type,
                lsr_rank=lsr_rank,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        """
        Args:
            idx: (B, T) token ids
            targets: optional (B, T) target token ids
        
        Returns:
            logits: (B, T, vocab_size)
            loss: scalar loss if targets provided, else None
        """
        B, T = idx.shape
        assert T <= self.max_seq_len

        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)

        for blk in self.blocks:
            x = blk(x)

        x = self.ln_f(x)
        logits = self.head(x)

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.reshape(-1),
                ignore_index=-100
            )
            return logits, loss
        else:
            return logits, None

