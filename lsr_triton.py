"""
Triton-optimized Low Separation Rank (LSR) Attention.

Fuses the low-rank projection and attention computation into efficient CUDA kernels.
"""

import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


# -----------------------------------------------------------------------------
# Triton Kernels
# -----------------------------------------------------------------------------

@triton.jit
def lsr_scores_kernel(
    Q_ptr, K_ptr, Wq_ptr, Wk_ptr, S_ptr,
    T: tl.constexpr, D: tl.constexpr, R: tl.constexpr,
    stride_q_bh, stride_q_t, stride_q_d,
    stride_k_bh, stride_k_t, stride_k_d,
    stride_wq_h, stride_wq_d, stride_wq_r,
    stride_wk_h, stride_wk_d, stride_wk_r,
    stride_s_bh, stride_s_i, stride_s_j,
    H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
    ORIGINAL_R: tl.constexpr,  # Original rank for proper scaling
):
    """
    Compute LSR attention scores: S[i,j] = sum_r (Q @ Wq)[i,r] * (K @ Wk)[j,r] / sqrt(R)
    
    Fuses the projection and score computation.
    """
    pid_m = tl.program_id(0)  # row block
    pid_n = tl.program_id(1)  # col block
    pid_bh = tl.program_id(2)  # batch * head
    
    # Extract head index for W lookup
    h = pid_bh % H
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    offs_r = tl.arange(0, BLOCK_R)
    
    m_mask = offs_m < T
    n_mask = offs_n < T
    
    # Load Q[bh, m, :] and K[bh, n, :]
    q_ptrs = Q_ptr + pid_bh * stride_q_bh + offs_m[:, None] * stride_q_t + offs_d[None, :] * stride_q_d
    k_ptrs = K_ptr + pid_bh * stride_k_bh + offs_n[:, None] * stride_k_t + offs_d[None, :] * stride_k_d
    
    Q_block = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)  # (BLOCK_M, D)
    K_block = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)  # (BLOCK_N, D)
    
    # Accumulate scores over rank dimension (only iterate over ORIGINAL_R, padded dims are zeros)
    scores = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for r_start in range(0, R, BLOCK_R):
        r_offs = r_start + offs_r
        r_mask = r_offs < ORIGINAL_R  # Use ORIGINAL_R to mask out padding
        
        # Load Wq[h, :, r] and Wk[h, :, r]
        wq_ptrs = Wq_ptr + h * stride_wq_h + offs_d[:, None] * stride_wq_d + r_offs[None, :] * stride_wq_r
        wk_ptrs = Wk_ptr + h * stride_wk_h + offs_d[:, None] * stride_wk_d + r_offs[None, :] * stride_wk_r
        
        Wq_block = tl.load(wq_ptrs, mask=r_mask[None, :], other=0.0)  # (D, BLOCK_R)
        Wk_block = tl.load(wk_ptrs, mask=r_mask[None, :], other=0.0)  # (D, BLOCK_R)
        
        # Project: Q_lr = Q @ Wq, K_lr = K @ Wk
        Q_lr = tl.dot(Q_block, Wq_block)  # (BLOCK_M, BLOCK_R)
        K_lr = tl.dot(K_block, Wk_block)  # (BLOCK_N, BLOCK_R)
        
        # Score contribution: sum over this rank block
        scores += tl.dot(Q_lr, tl.trans(K_lr))
    
    # Scale by 1/sqrt(ORIGINAL_R) - use original rank for proper scaling
    scale = 1.0 / tl.sqrt(tl.cast(ORIGINAL_R, tl.float32))
    scores = scores * scale
    
    # Store scores
    s_ptrs = S_ptr + pid_bh * stride_s_bh + offs_m[:, None] * stride_s_i + offs_n[None, :] * stride_s_j
    tl.store(s_ptrs, scores, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def lsr_softmax_kernel(
    S_ptr, P_ptr,
    T: tl.constexpr,
    stride_s_bh, stride_s_i, stride_s_j,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Apply causal mask and row-wise softmax."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < T
    
    # Load full row for softmax
    row_max = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    # First pass: find max
    for n_start in range(0, T, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < T
        
        s_ptrs = S_ptr + pid_bh * stride_s_bh + offs_m[:, None] * stride_s_i + offs_n[None, :] * stride_s_j
        s_block = tl.load(s_ptrs, mask=m_mask[:, None] & n_mask[None, :], other=float("-inf"))
        
        if CAUSAL:
            causal_mask = offs_n[None, :] <= offs_m[:, None]
            s_block = tl.where(causal_mask, s_block, float("-inf"))
        
        block_max = tl.max(s_block, axis=1)
        row_max = tl.maximum(row_max, block_max)
    
    # Second pass: compute exp sum
    for n_start in range(0, T, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < T
        
        s_ptrs = S_ptr + pid_bh * stride_s_bh + offs_m[:, None] * stride_s_i + offs_n[None, :] * stride_s_j
        s_block = tl.load(s_ptrs, mask=m_mask[:, None] & n_mask[None, :], other=float("-inf"))
        
        if CAUSAL:
            causal_mask = offs_n[None, :] <= offs_m[:, None]
            s_block = tl.where(causal_mask, s_block, float("-inf"))
        
        exp_block = tl.exp(s_block - row_max[:, None])
        row_sum += tl.sum(exp_block, axis=1)
    
    # Third pass: normalize and store
    for n_start in range(0, T, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < T
        
        s_ptrs = S_ptr + pid_bh * stride_s_bh + offs_m[:, None] * stride_s_i + offs_n[None, :] * stride_s_j
        p_ptrs = P_ptr + pid_bh * stride_s_bh + offs_m[:, None] * stride_s_i + offs_n[None, :] * stride_s_j
        
        s_block = tl.load(s_ptrs, mask=m_mask[:, None] & n_mask[None, :], other=float("-inf"))
        
        if CAUSAL:
            causal_mask = offs_n[None, :] <= offs_m[:, None]
            s_block = tl.where(causal_mask, s_block, float("-inf"))
        
        p_block = tl.exp(s_block - row_max[:, None]) / row_sum[:, None]
        p_block = tl.where(m_mask[:, None] & n_mask[None, :], p_block, 0.0)
        
        tl.store(p_ptrs, p_block, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def lsr_output_kernel(
    P_ptr, V_ptr, O_ptr,
    T: tl.constexpr, D: tl.constexpr,
    stride_p_bh, stride_p_i, stride_p_j,
    stride_v_bh, stride_v_t, stride_v_d,
    stride_o_bh, stride_o_t, stride_o_d,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute output: O = P @ V"""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    m_mask = offs_m < T
    d_mask = offs_d < D
    
    # Accumulate output
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    
    for n_start in range(0, T, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < T
        
        # Load P[m, n] and V[n, d]
        p_ptrs = P_ptr + pid_bh * stride_p_bh + offs_m[:, None] * stride_p_i + offs_n[None, :] * stride_p_j
        v_ptrs = V_ptr + pid_bh * stride_v_bh + offs_n[:, None] * stride_v_t + offs_d[None, :] * stride_v_d
        
        P_block = tl.load(p_ptrs, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
        V_block = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
        
        acc += tl.dot(P_block, V_block)
    
    # Store output
    o_ptrs = O_ptr + pid_bh * stride_o_bh + offs_m[:, None] * stride_o_t + offs_d[None, :] * stride_o_d
    tl.store(o_ptrs, acc, mask=m_mask[:, None] & d_mask[None, :])


# -----------------------------------------------------------------------------
# Python Wrappers
# -----------------------------------------------------------------------------

def lsr_attention_triton(q, k, v, W_q, W_k, causal=True):
    """
    Triton-optimized LSR attention.
    
    Args:
        q, k, v: (B, H, T, D)
        W_q, W_k: (H, D, R)
        causal: whether to apply causal masking
    
    Returns:
        output: (B, H, T, D)
    """
    B, H, T, D = q.shape
    R = W_q.shape[-1]
    BH = B * H
    
    # Triton requires K >= 16 for tl.dot, so pad R if needed
    R_padded = max(R, 16)
    if R < 16:
        pad_size = 16 - R
        W_q = torch.nn.functional.pad(W_q, (0, pad_size), value=0.0)
        W_k = torch.nn.functional.pad(W_k, (0, pad_size), value=0.0)
    
    # Flatten batch and head dimensions
    q_flat = q.reshape(BH, T, D).contiguous()
    k_flat = k.reshape(BH, T, D).contiguous()
    v_flat = v.reshape(BH, T, D).contiguous()
    
    # Allocate intermediates
    S = torch.empty(BH, T, T, device=q.device, dtype=torch.float32)
    P = torch.empty(BH, T, T, device=q.device, dtype=torch.float32)
    O = torch.empty(BH, T, D, device=q.device, dtype=torch.float32)
    
    # Kernel configs - use padded R for BLOCK_R
    BLOCK_M = min(64, T)
    BLOCK_N = min(64, T)
    BLOCK_R = min(32, R_padded)
    BLOCK_D = min(64, D)
    
    # Launch score kernel (use R_padded for dimensions, but original R for scaling)
    grid_scores = (triton.cdiv(T, BLOCK_M), triton.cdiv(T, BLOCK_N), BH)
    lsr_scores_kernel[grid_scores](
        q_flat, k_flat, W_q.contiguous(), W_k.contiguous(), S,
        T, D, R_padded,  # Use padded R for kernel dimensions
        *q_flat.stride(), *k_flat.stride(),
        *W_q.stride(), *W_k.stride(),
        *S.stride(),
        H,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_R=BLOCK_R,
        ORIGINAL_R=R,  # Pass original R for proper scaling
    )
    
    # Launch softmax kernel
    grid_softmax = (triton.cdiv(T, BLOCK_M), BH)
    lsr_softmax_kernel[grid_softmax](
        S, P,
        T,
        *S.stride(),
        CAUSAL=causal,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    
    # Launch output kernel
    grid_output = (triton.cdiv(T, BLOCK_M), BH)
    lsr_output_kernel[grid_output](
        P, v_flat, O,
        T, D,
        *P.stride(), *v_flat.stride(), *O.stride(),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
    )
    
    return O.reshape(B, H, T, D).to(q.dtype)


# -----------------------------------------------------------------------------
# Autograd Wrapper
# -----------------------------------------------------------------------------

class LSRTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, W_q, W_k, causal):
        out = lsr_attention_triton(q.float(), k.float(), v.float(), 
                                    W_q.float(), W_k.float(), causal)
        ctx.save_for_backward(q, k, v, W_q, W_k)
        ctx.causal = causal
        return out.to(q.dtype)
    
    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, W_q, W_k = ctx.saved_tensors
        causal = ctx.causal
        
        # Fall back to PyTorch autograd for backward
        q = q.detach().requires_grad_(True)
        k = k.detach().requires_grad_(True)
        v = v.detach().requires_grad_(True)
        W_q = W_q.detach().requires_grad_(True)
        W_k = W_k.detach().requires_grad_(True)
        
        with torch.enable_grad():
            # Recompute forward with autograd
            B, H, T, D = q.shape
            R = W_q.shape[-1]
            
            q_lr = torch.einsum("bhtd,hdr->bhtr", q, W_q)
            k_lr = torch.einsum("bhtd,hdr->bhtr", k, W_k)
            
            scale = 1.0 / math.sqrt(max(R, 1))
            scores = torch.einsum("bhir,bhjr->bhij", q_lr, k_lr) * scale
            
            if causal:
                causal_mask = torch.triu(
                    torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1
                )
                scores = scores.masked_fill(causal_mask, float("-inf"))
            
            attn = torch.softmax(scores, dim=-1)
            out = torch.einsum("bhij,bhjd->bhid", attn, v)
            
            out.backward(grad_output)
        
        return q.grad, k.grad, v.grad, W_q.grad, W_k.grad, None


def lsr_attention_triton_autograd(q, k, v, W_q, W_k, causal=True):
    """LSR attention with Triton forward and autograd backward."""
    return LSRTritonFunction.apply(q, k, v, W_q, W_k, causal)


# -----------------------------------------------------------------------------
# Module
# -----------------------------------------------------------------------------

class MultiHeadLSRAttentionTriton(nn.Module):
    """
    Multi-head LSR attention using Triton kernels.
    
    Args:
        d_model: model dimension
        num_heads: number of attention heads
        lsr_rank: rank for LSR attention
    """
    
    def __init__(self, d_model, num_heads, lsr_rank=32):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.lsr_rank = lsr_rank

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self.W_q_lsr = nn.Parameter(
            torch.randn(num_heads, self.d_head, lsr_rank) / math.sqrt(self.d_head)
        )
        self.W_k_lsr = nn.Parameter(
            torch.randn(num_heads, self.d_head, lsr_rank) / math.sqrt(self.d_head)
        )

    def forward(self, x):
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)

        y = lsr_attention_triton_autograd(q, k, v, self.W_q_lsr, self.W_k_lsr, causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.o_proj(y)


# -----------------------------------------------------------------------------
# Test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import time
    
    torch.manual_seed(42)
    device = "cuda"
    
    B, H, T, D, R = 4, 8, 1024, 64, 32
    
    q = torch.randn(B, H, T, D, device=device, dtype=torch.float32)
    k = torch.randn(B, H, T, D, device=device, dtype=torch.float32)
    v = torch.randn(B, H, T, D, device=device, dtype=torch.float32)
    W_q = torch.randn(H, D, R, device=device, dtype=torch.float32) / math.sqrt(D)
    W_k = torch.randn(H, D, R, device=device, dtype=torch.float32) / math.sqrt(D)
    
    # Reference implementation
    def lsr_reference(q, k, v, W_q, W_k):
        q_lr = torch.einsum("bhtd,hdr->bhtr", q, W_q)
        k_lr = torch.einsum("bhtd,hdr->bhtr", k, W_k)
        scores = torch.einsum("bhir,bhjr->bhij", q_lr, k_lr) / math.sqrt(R)
        causal_mask = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        return torch.einsum("bhij,bhjd->bhid", attn, v)
    
    # Warmup
    for _ in range(3):
        _ = lsr_attention_triton(q, k, v, W_q, W_k, causal=True)
        _ = lsr_reference(q, k, v, W_q, W_k)
    torch.cuda.synchronize()
    
    # Correctness
    out_triton = lsr_attention_triton(q, k, v, W_q, W_k, causal=True)
    out_ref = lsr_reference(q, k, v, W_q, W_k)
    print(f"Max diff: {(out_triton - out_ref).abs().max().item():.6f}")
    
    # Benchmark
    iters = 50
    
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        _ = lsr_attention_triton(q, k, v, W_q, W_k, causal=True)
    torch.cuda.synchronize()
    triton_time = (time.time() - t0) / iters * 1000
    
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        _ = lsr_reference(q, k, v, W_q, W_k)
    torch.cuda.synchronize()
    ref_time = (time.time() - t0) / iters * 1000
    
    print(f"Triton LSR: {triton_time:.2f} ms")
    print(f"PyTorch LSR: {ref_time:.2f} ms")
    print(f"Speedup: {ref_time/triton_time:.2f}x")

