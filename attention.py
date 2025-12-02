"""Attention implementations: Scaled Dot-Product and LSR with Triton."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from lsr_triton_parallel import lsr_attention_online, lsr_attention_fused


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

    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        with sdpa_kernel([SDPBackend.MATH]):
            out = F.scaled_dot_product_attention(
                q_flat, k_flat, v_flat,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=causal,
            )
    except ImportError:
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            out = F.scaled_dot_product_attention(
                q_flat, k_flat, v_flat,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=causal,
            )
    return out.view(B, H, T, D)


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention with support for standard and LSR attention.

    Args:
        d_model: model dimension
        num_heads: number of attention heads
        attn_type: "dot" for scaled dot-product, "lsr" for Triton-optimized LSR,
                   "lsr_fused" for fused QKV+attention kernel
        lsr_rank: rank for LSR attention (only used if attn_type == "lsr" or "lsr_fused")
    """

    def __init__(self, d_model, num_heads, attn_type="dot", lsr_rank=32):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.attn_type = attn_type
        self.lsr_rank = lsr_rank

        if attn_type == "lsr_fused":
            # Fused QKV projection: (d_model, 3*d_model)
            self.W_qkv = nn.Parameter(torch.randn(d_model, 3 * d_model) / math.sqrt(d_model))
            self.W_o = nn.Parameter(torch.randn(d_model, d_model) / math.sqrt(d_model))
            self.W_q_lsr = nn.Parameter(
                torch.randn(num_heads, self.d_head, lsr_rank) / math.sqrt(self.d_head)
            )
            self.W_k_lsr = nn.Parameter(
                torch.randn(num_heads, self.d_head, lsr_rank) / math.sqrt(self.d_head)
            )
            self.lsr_core = nn.Parameter(torch.ones(num_heads, lsr_rank))
            # Register None for unfused params
            self.q_proj = None
            self.k_proj = None
            self.v_proj = None
            self.o_proj = None
        else:
            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
            self.o_proj = nn.Linear(d_model, d_model)
            self.W_qkv = None
            self.W_o = None

            if attn_type == "lsr":
                self.W_q_lsr = nn.Parameter(
                    torch.randn(num_heads, self.d_head, lsr_rank) / math.sqrt(self.d_head)
                )
                self.W_k_lsr = nn.Parameter(
                    torch.randn(num_heads, self.d_head, lsr_rank) / math.sqrt(self.d_head)
                )
                self.lsr_core = nn.Parameter(torch.ones(num_heads, lsr_rank))
            else:
                self.W_q_lsr = None
                self.W_k_lsr = None
                self.lsr_core = None

    def forward(self, x):
        """
        Args:
            x: (B, T, d_model) input tensor

        Returns:
            output: (B, T, d_model)
        """
        B, T, _ = x.shape

        if self.attn_type == "lsr_fused":
            # Single fused kernel for QKV projection + attention + output projection
            return lsr_attention_fused(
                x, self.W_qkv, self.W_q_lsr, self.W_k_lsr, self.lsr_core, self.W_o, causal=True
            )

        q = self.q_proj(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)

        if self.attn_type == "lsr":
            y = lsr_attention_online(
                q, k, v, self.W_q_lsr, self.W_k_lsr, self.lsr_core, causal=True
            )
        else:
            y = scaled_dot_product_attention(q, k, v, causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.o_proj(y)
