"""Attention implementations: Scaled Dot-Product and Low Separation Rank."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v, causal=True):
    """
    Standard scaled dot-product attention using PyTorch's optimized implementation.
    
    Args:
        q, k, v: (B, H, T, D) query, key, value tensors
        causal: whether to apply causal masking
    
    Returns:
        output: (B, H, T, D)
    """
    B, H, T, D = q.shape
    
    q_flat = q.reshape(B * H, T, D)
    k_flat = k.reshape(B * H, T, D)
    v_flat = v.reshape(B * H, T, D)

    out = F.scaled_dot_product_attention(
        q_flat, k_flat, v_flat,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=causal,
    )
    return out.view(B, H, T, D)


def lsr_attention(q, k, v, W_q, W_k, causal=True):
    """
    Low Separation Rank (LSR) attention.
    
    Projects Q and K to a lower-rank space before computing attention scores,
    resulting in attention matrices with bounded separation rank.
    
    Args:
        q, k, v: (B, H, T, D) query, key, value tensors
        W_q: (H, D, R) per-head query projection to rank-R space
        W_k: (H, D, R) per-head key projection to rank-R space
        causal: whether to apply causal masking
    
    Returns:
        output: (B, H, T, D)
    """
    B, H, T, D = q.shape
    R = W_q.shape[-1]
    
    # Project to low-rank space: (B, H, T, R)
    q_lr = torch.einsum("bhtd,hdr->bhtr", q, W_q)
    k_lr = torch.einsum("bhtd,hdr->bhtr", k, W_k)
    
    # Compute scores in low-rank space
    scale = 1.0 / math.sqrt(max(R, 1))
    scores = torch.einsum("bhir,bhjr->bhij", q_lr, k_lr) * scale
    
    # Apply causal mask
    if causal:
        causal_mask = torch.triu(
            torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))
    
    # Softmax and apply to values
    attn = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhij,bhjd->bhid", attn, v)
    
    return out


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention with support for both standard and LSR attention.
    
    Args:
        d_model: model dimension
        num_heads: number of attention heads
        attn_type: "dot" for scaled dot-product, "lsr" for low separation rank
        lsr_rank: rank for LSR attention (only used if attn_type="lsr")
    """
    
    def __init__(self, d_model, num_heads, attn_type="dot", lsr_rank=32):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.attn_type = attn_type
        self.lsr_rank = lsr_rank

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        if attn_type == "lsr":
            self.W_q_lsr = nn.Parameter(
                torch.randn(num_heads, self.d_head, lsr_rank) / math.sqrt(self.d_head)
            )
            self.W_k_lsr = nn.Parameter(
                torch.randn(num_heads, self.d_head, lsr_rank) / math.sqrt(self.d_head)
            )
        else:
            self.W_q_lsr = None
            self.W_k_lsr = None

    def forward(self, x):
        """
        Args:
            x: (B, T, d_model) input tensor
        
        Returns:
            output: (B, T, d_model)
        """
        B, T, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)

        # Apply attention
        if self.attn_type == "lsr":
            y = lsr_attention(q, k, v, self.W_q_lsr, self.W_k_lsr, causal=True)
        else:
            y = scaled_dot_product_attention(q, k, v, causal=True)

        # Merge heads and project output
        y = y.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.o_proj(y)

