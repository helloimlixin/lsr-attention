"""Optimized Triton LSR attention kernels.

Features:
- Online softmax (single-pass FlashAttention-style)
- Kronecker factorized core with compile-time R1 unrolling
- Autotuned block sizes for optimal performance
- Fused QKV+attention kernel for maximum performance
"""

import math
import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.runtime.errors import OutOfResources


# -----------------------------------------------------------------------------
# Fused QKV Projection + LSR Attention Kernel
# -----------------------------------------------------------------------------

_LSR_FUSED_FALLBACK_WARNED = False

@triton.jit
def lsr_fused_qkvo_kernel(
    # Input
    X_ptr,
    # QKV projection weights (fused: d_model x 3*d_model)
    W_qkv_ptr,
    # LSR projection weights
    Wq_lsr_ptr, Wk_lsr_ptr, Core_ptr,
    # Output projection weight
    W_o_ptr,
    # Output
    O_ptr,
    # Dimensions
    B: tl.constexpr, T: tl.constexpr, D_MODEL: tl.constexpr,
    H: tl.constexpr, D_HEAD: tl.constexpr, R: tl.constexpr,
    # Strides for X: (B, T, D_MODEL)
    stride_x_b, stride_x_t, stride_x_d,
    # Strides for W_qkv: (D_MODEL, 3*D_MODEL)
    stride_wqkv_in, stride_wqkv_out,
    # Strides for LSR weights: (H, D_HEAD, R)
    stride_wlsr_h, stride_wlsr_d, stride_wlsr_r,
    # Strides for core: (H, R)
    stride_core_h, stride_core_r,
    # Strides for W_o: (D_MODEL, D_MODEL)
    stride_wo_in, stride_wo_out,
    # Strides for O: (B, T, D_MODEL)
    stride_o_b, stride_o_t, stride_o_d,
    # Config
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D_HEAD: tl.constexpr,
    BLOCK_R: tl.constexpr,
    ORIGINAL_R: tl.constexpr,
    USE_FP16: tl.constexpr,
):
    """Fused QKV projection + LSR attention + output projection kernel.

    This kernel fuses:
    1. Q = X @ W_q, K = X @ W_k, V = X @ W_v (via fused W_qkv)
    2. LSR attention computation
    3. Output = attn_out @ W_o

    Into a single kernel launch, eliminating memory round-trips.
    """
    pid_m = tl.program_id(0)  # Which block of queries
    pid_b = tl.program_id(1)  # Which batch item

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d_head = tl.arange(0, BLOCK_D_HEAD)
    offs_r = tl.arange(0, BLOCK_R)
    offs_d_model = tl.arange(0, D_MODEL)

    m_mask = offs_m < T
    d_head_mask = offs_d_head < D_HEAD
    r_mask = offs_r < ORIGINAL_R

    # Accumulated output across all heads (d_model dimension)
    O_acc = tl.zeros((BLOCK_M, D_MODEL), dtype=tl.float32)

    # Process each head
    for h in range(H):
        head_start = h * D_HEAD

        # Load X block for query positions: (BLOCK_M, D_MODEL)
        x_q_ptrs = X_ptr + pid_b * stride_x_b + offs_m[:, None] * stride_x_t + offs_d_model[None, :] * stride_x_d
        X_q_block = tl.load(x_q_ptrs, mask=m_mask[:, None], other=0.0)

        # Load W_q slice: (D_MODEL, D_HEAD) - columns [head_start:head_start+D_HEAD]
        wq_ptrs = W_qkv_ptr + offs_d_model[:, None] * stride_wqkv_in + (head_start + offs_d_head[None, :]) * stride_wqkv_out
        W_q_slice = tl.load(wq_ptrs, mask=d_head_mask[None, :], other=0.0)

        # Compute Q for this head: (BLOCK_M, D_HEAD)
        if USE_FP16:
            X_q_16 = X_q_block.to(tl.float16)
            W_q_16 = W_q_slice.to(tl.float16)
            Q_block = tl.dot(X_q_16, W_q_16).to(tl.float32)
        else:
            Q_block = tl.dot(X_q_block, W_q_slice)

        # Load LSR projection for Q: (D_HEAD, R)
        wq_lsr_ptrs = Wq_lsr_ptr + h * stride_wlsr_h + offs_d_head[:, None] * stride_wlsr_d + offs_r[None, :] * stride_wlsr_r
        Wq_lsr = tl.load(wq_lsr_ptrs, mask=d_head_mask[:, None] & r_mask[None, :], other=0.0)

        # Load core for this head: (R,)
        core_ptrs = Core_ptr + h * stride_core_h + offs_r * stride_core_r
        core_block = tl.load(core_ptrs, mask=r_mask, other=0.0)

        # Q low-rank: (BLOCK_M, R)
        if USE_FP16:
            Q_block_16 = Q_block.to(tl.float16)
            Wq_lsr_16 = Wq_lsr.to(tl.float16)
            Q_lr = tl.dot(Q_block_16, Wq_lsr_16).to(tl.float32)
        else:
            Q_lr = tl.dot(Q_block, Wq_lsr)

        scale = 1.0 / tl.sqrt(tl.cast(ORIGINAL_R, tl.float32))
        Q_lr = Q_lr * core_block[None, :] * scale

        # Load Wk_lsr for this head
        wk_lsr_ptrs = Wk_lsr_ptr + h * stride_wlsr_h + offs_d_head[:, None] * stride_wlsr_d + offs_r[None, :] * stride_wlsr_r
        Wk_lsr = tl.load(wk_lsr_ptrs, mask=d_head_mask[:, None] & r_mask[None, :], other=0.0)

        # Load W_k slice and W_v slice for this head
        wk_ptrs = W_qkv_ptr + offs_d_model[:, None] * stride_wqkv_in + (D_MODEL + head_start + offs_d_head[None, :]) * stride_wqkv_out
        W_k_slice = tl.load(wk_ptrs, mask=d_head_mask[None, :], other=0.0)

        wv_ptrs = W_qkv_ptr + offs_d_model[:, None] * stride_wqkv_in + (2 * D_MODEL + head_start + offs_d_head[None, :]) * stride_wqkv_out
        W_v_slice = tl.load(wv_ptrs, mask=d_head_mask[None, :], other=0.0)

        if CAUSAL:
            n_blocks = tl.minimum(tl.cdiv((pid_m + 1) * BLOCK_M, BLOCK_N), tl.cdiv(T, BLOCK_N))
        else:
            n_blocks = tl.cdiv(T, BLOCK_N)

        # Online softmax state for this head
        m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D_HEAD), dtype=tl.float32)

        for n_block in range(n_blocks):
            offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
            n_mask = offs_n < T

            # Load X block for key/value positions: (BLOCK_N, D_MODEL)
            x_kv_ptrs = X_ptr + pid_b * stride_x_b + offs_n[:, None] * stride_x_t + offs_d_model[None, :] * stride_x_d
            X_kv_block = tl.load(x_kv_ptrs, mask=n_mask[:, None], other=0.0)

            # Compute K for this head: (BLOCK_N, D_HEAD)
            if USE_FP16:
                X_kv_16 = X_kv_block.to(tl.float16)
                W_k_16 = W_k_slice.to(tl.float16)
                K_block = tl.dot(X_kv_16, W_k_16).to(tl.float32)
            else:
                K_block = tl.dot(X_kv_block, W_k_slice)

            # K low-rank: (BLOCK_N, R)
            if USE_FP16:
                K_block_16 = K_block.to(tl.float16)
                Wk_lsr_16 = Wk_lsr.to(tl.float16)
                K_lr = tl.dot(K_block_16, Wk_lsr_16).to(tl.float32)
            else:
                K_lr = tl.dot(K_block, Wk_lsr)

            # Scores: Q_lr @ K_lr^T
            if USE_FP16:
                Q_lr_16 = Q_lr.to(tl.float16)
                K_lr_16 = K_lr.to(tl.float16)
                scores = tl.dot(Q_lr_16, tl.trans(K_lr_16)).to(tl.float32)
            else:
                scores = tl.dot(Q_lr, tl.trans(K_lr))

            if CAUSAL:
                causal_mask = offs_n[None, :] <= offs_m[:, None]
                scores = tl.where(causal_mask, scores, float("-inf"))
            scores = tl.where(n_mask[None, :], scores, float("-inf"))

            # Online softmax update
            m_ij = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, m_ij)

            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha
            acc = acc * alpha[:, None]

            p_ij = tl.exp(scores - m_new[:, None])
            l_i = l_i + tl.sum(p_ij, axis=1)

            # Compute V for this head: (BLOCK_N, D_HEAD)
            if USE_FP16:
                W_v_16 = W_v_slice.to(tl.float16)
                V_block = tl.dot(X_kv_16, W_v_16).to(tl.float32)
            else:
                V_block = tl.dot(X_kv_block, W_v_slice)

            if USE_FP16:
                p_ij_16 = p_ij.to(tl.float16)
                V_block_16 = V_block.to(tl.float16)
                acc = acc + tl.dot(p_ij_16, V_block_16).to(tl.float32)
            else:
                acc = acc + tl.dot(p_ij.to(V_block.dtype), V_block)

            m_i = m_new

        # Finalize attention output for this head: (BLOCK_M, D_HEAD)
        attn_out = acc / l_i[:, None]

        # Apply output projection slice for this head
        # W_o slice: (D_HEAD, D_MODEL) - rows [head_start:head_start+D_HEAD]
        wo_ptrs = W_o_ptr + (head_start + offs_d_head[:, None]) * stride_wo_in + offs_d_model[None, :] * stride_wo_out
        W_o_slice = tl.load(wo_ptrs, mask=d_head_mask[:, None], other=0.0)

        if USE_FP16:
            attn_out_16 = attn_out.to(tl.float16)
            W_o_slice_16 = W_o_slice.to(tl.float16)
            O_acc = O_acc + tl.dot(attn_out_16, W_o_slice_16).to(tl.float32)
        else:
            O_acc = O_acc + tl.dot(attn_out, W_o_slice)

    # Store output
    o_ptrs = O_ptr + pid_b * stride_o_b + offs_m[:, None] * stride_o_t + offs_d_model[None, :] * stride_o_d
    tl.store(o_ptrs, O_acc, mask=m_mask[:, None])


# -----------------------------------------------------------------------------
# Online Softmax Kernel (Flat Core) - Autotuned
# -----------------------------------------------------------------------------

@triton.autotune(
    configs=[
        # High throughput configs for larger sequences
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        # Balanced configs
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        # Small block configs for shorter sequences
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4, num_stages=4),
    ],
    key=['T', 'D', 'ORIGINAL_R'],
)
@triton.jit
def lsr_online_kernel(
    Q_ptr, K_ptr, V_ptr, Wq_ptr, Wk_ptr, Core_ptr, O_ptr,
    T: tl.constexpr, D: tl.constexpr, R: tl.constexpr,
    stride_q_bh, stride_q_t, stride_q_d,
    stride_k_bh, stride_k_t, stride_k_d,
    stride_v_bh, stride_v_t, stride_v_d,
    stride_wq_h, stride_wq_d, stride_wq_r,
    stride_wk_h, stride_wk_d, stride_wk_r,
    stride_core_h, stride_core_r,
    stride_o_bh, stride_o_t, stride_o_d,
    H: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ORIGINAL_R: tl.constexpr,
    USE_FP16: tl.constexpr,
):
    """LSR attention with online softmax (single-pass).

    When USE_FP16=True, uses FP16 for matrix multiplications (tensor cores)
    while keeping accumulators in FP32 for numerical stability.
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_r = tl.arange(0, BLOCK_R)

    m_mask = offs_m < T
    d_mask = offs_d < D
    r_mask = offs_r < ORIGINAL_R

    # Load projection matrices and core (always in FP32 for accuracy)
    wq_ptrs = Wq_ptr + h * stride_wq_h + offs_d[:, None] * stride_wq_d + offs_r[None, :] * stride_wq_r
    wk_ptrs = Wk_ptr + h * stride_wk_h + offs_d[:, None] * stride_wk_d + offs_r[None, :] * stride_wk_r
    Wq_block = tl.load(wq_ptrs, mask=d_mask[:, None] & r_mask[None, :], other=0.0)
    Wk_block = tl.load(wk_ptrs, mask=d_mask[:, None] & r_mask[None, :], other=0.0)

    core_ptrs = Core_ptr + h * stride_core_h + offs_r * stride_core_r
    core_block = tl.load(core_ptrs, mask=r_mask, other=0.0)

    # Load Q block and project
    q_ptrs = Q_ptr + pid_bh * stride_q_bh + offs_m[:, None] * stride_q_t + offs_d[None, :] * stride_q_d
    Q_block = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)

    # Convert to FP16 for faster matmul if enabled
    if USE_FP16:
        Q_block_mm = Q_block.to(tl.float16)
        Wq_block_mm = Wq_block.to(tl.float16)
        Wk_block_mm = Wk_block.to(tl.float16)
        Q_lr = tl.dot(Q_block_mm, Wq_block_mm).to(tl.float32)
    else:
        Q_lr = tl.dot(Q_block, Wq_block)

    scale = 1.0 / tl.sqrt(tl.cast(ORIGINAL_R, tl.float32))
    Q_lr = Q_lr * core_block[None, :] * scale

    if CAUSAL:
        n_blocks = tl.minimum(tl.cdiv((pid_m + 1) * BLOCK_M, BLOCK_N), tl.cdiv(T, BLOCK_N))
    else:
        n_blocks = tl.cdiv(T, BLOCK_N)

    # Online softmax state (always FP32)
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for n_block in range(n_blocks):
        offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < T

        # Load K block and project
        k_ptrs = K_ptr + pid_bh * stride_k_bh + offs_n[:, None] * stride_k_t + offs_d[None, :] * stride_k_d
        K_block = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)

        if USE_FP16:
            K_block_mm = K_block.to(tl.float16)
            K_lr = tl.dot(K_block_mm, Wk_block_mm).to(tl.float32)
            # Score computation in FP16 for tensor cores
            Q_lr_16 = Q_lr.to(tl.float16)
            K_lr_16 = K_lr.to(tl.float16)
            scores = tl.dot(Q_lr_16, tl.trans(K_lr_16)).to(tl.float32)
        else:
            K_lr = tl.dot(K_block, Wk_block)
            scores = tl.dot(Q_lr, tl.trans(K_lr))

        if CAUSAL:
            causal_mask = offs_n[None, :] <= offs_m[:, None]
            scores = tl.where(causal_mask, scores, float("-inf"))
        scores = tl.where(n_mask[None, :], scores, float("-inf"))

        # Online softmax update (FP32)
        m_ij = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha
        acc = acc * alpha[:, None]

        p_ij = tl.exp(scores - m_new[:, None])
        l_i = l_i + tl.sum(p_ij, axis=1)

        v_ptrs = V_ptr + pid_bh * stride_v_bh + offs_n[:, None] * stride_v_t + offs_d[None, :] * stride_v_d
        V_block = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)

        if USE_FP16:
            p_ij_16 = p_ij.to(tl.float16)
            V_block_16 = V_block.to(tl.float16)
            acc = acc + tl.dot(p_ij_16, V_block_16).to(tl.float32)
        else:
            acc = acc + tl.dot(p_ij.to(V_block.dtype), V_block)

        m_i = m_new

    acc = acc / l_i[:, None]

    o_ptrs = O_ptr + pid_bh * stride_o_bh + offs_m[:, None] * stride_o_t + offs_d[None, :] * stride_o_d
    tl.store(o_ptrs, acc, mask=m_mask[:, None] & d_mask[None, :])


# -----------------------------------------------------------------------------
# Kronecker Kernel with static_range unrolling - Autotuned
# -----------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=4),
    ],
    key=['T', 'D', 'R1', 'R2'],
)
@triton.jit
def lsr_kronecker_kernel(
    Q_ptr, K_ptr, V_ptr, Wq_ptr, Wk_ptr, Core1_ptr, Core2_ptr, O_ptr,
    T: tl.constexpr, D: tl.constexpr,
    R1: tl.constexpr, R2: tl.constexpr,
    stride_q_bh, stride_q_t, stride_q_d,
    stride_k_bh, stride_k_t, stride_k_d,
    stride_v_bh, stride_v_t, stride_v_d,
    stride_wq_h, stride_wq_d, stride_wq_r,
    stride_wk_h, stride_wk_d, stride_wk_r,
    stride_core1_h, stride_core1_r,
    stride_core2_h, stride_core2_r,
    stride_o_bh, stride_o_t, stride_o_d,
    H: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R2: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Kronecker LSR attention with online softmax and compile-time R1 unrolling."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_r2 = tl.arange(0, BLOCK_R2)

    m_mask = offs_m < T
    d_mask = offs_d < D
    r2_mask = offs_r2 < R2

    # Load core2
    core2_ptrs = Core2_ptr + h * stride_core2_h + offs_r2 * stride_core2_r
    core2 = tl.load(core2_ptrs, mask=r2_mask, other=0.0)

    # Load Q block
    q_ptrs = Q_ptr + pid_bh * stride_q_bh + offs_m[:, None] * stride_q_t + offs_d[None, :] * stride_q_d
    Q_block = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)

    scale = 1.0 / tl.sqrt(tl.cast(R1 * R2, tl.float32))

    if CAUSAL:
        n_blocks = tl.minimum(tl.cdiv((pid_m + 1) * BLOCK_M, BLOCK_N), tl.cdiv(T, BLOCK_N))
    else:
        n_blocks = tl.cdiv(T, BLOCK_N)

    # Online softmax state
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for n_block in range(n_blocks):
        offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < T

        k_ptrs = K_ptr + pid_bh * stride_k_bh + offs_n[:, None] * stride_k_t + offs_d[None, :] * stride_k_d
        K_block = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)

        # Accumulate scores across R1 factors (compile-time unroll)
        scores = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for r1 in tl.static_range(R1):
            core1_val = tl.load(Core1_ptr + h * stride_core1_h + r1 * stride_core1_r)

            wq_ptrs = Wq_ptr + h * stride_wq_h + offs_d[:, None] * stride_wq_d + (r1 * R2 + offs_r2[None, :]) * stride_wq_r
            wk_ptrs = Wk_ptr + h * stride_wk_h + offs_d[:, None] * stride_wk_d + (r1 * R2 + offs_r2[None, :]) * stride_wk_r
            Wq_block = tl.load(wq_ptrs, mask=d_mask[:, None] & r2_mask[None, :], other=0.0)
            Wk_block = tl.load(wk_ptrs, mask=d_mask[:, None] & r2_mask[None, :], other=0.0)

            Q_lr = tl.dot(Q_block, Wq_block) * core2[None, :]
            K_lr = tl.dot(K_block, Wk_block)

            scores = scores + core1_val * tl.dot(Q_lr, tl.trans(K_lr))

        scores = scores * scale

        if CAUSAL:
            causal_mask = offs_n[None, :] <= offs_m[:, None]
            scores = tl.where(causal_mask, scores, float("-inf"))
        scores = tl.where(n_mask[None, :], scores, float("-inf"))

        # Online softmax update
        m_ij = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha
        acc = acc * alpha[:, None]

        p_ij = tl.exp(scores - m_new[:, None])
        l_i = l_i + tl.sum(p_ij, axis=1)

        v_ptrs = V_ptr + pid_bh * stride_v_bh + offs_n[:, None] * stride_v_t + offs_d[None, :] * stride_v_d
        V_block = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
        acc = acc + tl.dot(p_ij.to(V_block.dtype), V_block)

        m_i = m_new

    acc = acc / l_i[:, None]

    o_ptrs = O_ptr + pid_bh * stride_o_bh + offs_m[:, None] * stride_o_t + offs_d[None, :] * stride_o_d
    tl.store(o_ptrs, acc, mask=m_mask[:, None] & d_mask[None, :])


# -----------------------------------------------------------------------------
# Python wrappers
# -----------------------------------------------------------------------------

def _run_lsr_fused_qkvo(x, W_qkv, W_q_lsr, W_k_lsr, core, W_o, causal=True, use_fp16=True):
    """Run fused QKV projection + LSR attention + output projection.

    Args:
        x: Input tensor (B, T, d_model)
        W_qkv: Fused QKV projection weights (d_model, 3*d_model)
        W_q_lsr: LSR Q projection (H, D_head, R)
        W_k_lsr: LSR K projection (H, D_head, R)
        core: LSR core (H, R)
        W_o: Output projection weights (d_model, d_model)
        causal: Whether to use causal masking
        use_fp16: Whether to use FP16 for matrix multiplications
    """
    B, T, D_MODEL = x.shape
    H = W_q_lsr.shape[0]
    D_HEAD = W_q_lsr.shape[1]
    R = W_q_lsr.shape[2]

    # Pad R to power of 2 >= 16 for efficient matrix ops
    R_padded = max(16, triton.next_power_of_2(R))
    if R_padded != R:
        pad = R_padded - R
        W_q_lsr = torch.nn.functional.pad(W_q_lsr, (0, pad), value=0.0)
        W_k_lsr = torch.nn.functional.pad(W_k_lsr, (0, pad), value=0.0)
        core = torch.nn.functional.pad(core, (0, pad), value=0.0)

    O = torch.empty(B, T, D_MODEL, device=x.device, dtype=torch.float32)

    BLOCK_D_HEAD = triton.next_power_of_2(D_HEAD)
    BLOCK_R = min(32, R_padded)
    BLOCK_M = 64
    BLOCK_N = 64

    grid = (triton.cdiv(T, BLOCK_M), B)

    lsr_fused_qkvo_kernel[grid](
        x.contiguous(),
        W_qkv.contiguous(),
        W_q_lsr.contiguous(), W_k_lsr.contiguous(), core.contiguous(),
        W_o.contiguous(),
        O,
        B, T, D_MODEL, H, D_HEAD, R_padded,
        *x.stride(),
        *W_qkv.stride(),
        *W_q_lsr.stride(),
        *core.stride(),
        *W_o.stride(),
        *O.stride(),
        CAUSAL=causal,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D_HEAD=BLOCK_D_HEAD,
        BLOCK_R=BLOCK_R,
        ORIGINAL_R=R,
        USE_FP16=use_fp16,
    )
    return O.to(x.dtype)


def _run_lsr_online(q, k, v, W_q, W_k, core, causal=True, use_fp16=True):
    """Run LSR attention with online softmax.

    Args:
        use_fp16: If True, uses FP16 for matrix multiplications (tensor cores).
                  Accumulators stay in FP32 for numerical stability.
    """
    B, H, T, D = q.shape
    R = W_q.shape[-1]
    BH = B * H

    # Pad R to power of 2 >= 16 for efficient matrix ops
    R_padded = max(16, triton.next_power_of_2(R))
    if R_padded != R:
        pad = R_padded - R
        W_q = torch.nn.functional.pad(W_q, (0, pad), value=0.0)
        W_k = torch.nn.functional.pad(W_k, (0, pad), value=0.0)
        core = torch.nn.functional.pad(core, (0, pad), value=0.0)

    q_flat = q.reshape(BH, T, D).contiguous()
    k_flat = k.reshape(BH, T, D).contiguous()
    v_flat = v.reshape(BH, T, D).contiguous()

    O = torch.empty(BH, T, D, device=q.device, dtype=torch.float32)

    BLOCK_R = min(32, R_padded)
    BLOCK_D = triton.next_power_of_2(D)

    # Grid uses lambda for autotune compatibility - BLOCK_M determined at runtime
    def grid(meta):
        return (triton.cdiv(T, meta['BLOCK_M']), BH)

    lsr_online_kernel[grid](
        q_flat, k_flat, v_flat,
        W_q.contiguous(), W_k.contiguous(), core.contiguous(),
        O,
        T, D, R_padded,
        *q_flat.stride(), *k_flat.stride(), *v_flat.stride(),
        *W_q.stride(), *W_k.stride(),
        *core.stride(),
        *O.stride(),
        H,
        CAUSAL=causal,
        BLOCK_R=BLOCK_R, BLOCK_D=BLOCK_D,
        ORIGINAL_R=R,
        USE_FP16=use_fp16,
    )
    return O.reshape(B, H, T, D).to(q.dtype)


def _run_lsr_kronecker(q, k, v, W_q, W_k, core1, core2, causal=True):
    """Run Kronecker LSR attention."""
    B, H, T, D = q.shape
    R1 = core1.shape[-1]
    R2 = core2.shape[-1]
    BH = B * H

    BLOCK_R2 = max(16, triton.next_power_of_2(R2))
    R2_padded = BLOCK_R2

    if R2_padded != R2:
        pad = R2_padded - R2
        core2 = torch.nn.functional.pad(core2, (0, pad), value=0.0)
        W_q = torch.nn.functional.pad(W_q, (0, pad * R1), value=0.0)
        W_k = torch.nn.functional.pad(W_k, (0, pad * R1), value=0.0)

    q_flat = q.reshape(BH, T, D).contiguous()
    k_flat = k.reshape(BH, T, D).contiguous()
    v_flat = v.reshape(BH, T, D).contiguous()

    O = torch.empty(BH, T, D, device=q.device, dtype=torch.float32)

    BLOCK_D = triton.next_power_of_2(D)

    # Grid uses lambda for autotune compatibility
    def grid(meta):
        return (triton.cdiv(T, meta['BLOCK_M']), BH)

    lsr_kronecker_kernel[grid](
        q_flat, k_flat, v_flat,
        W_q.contiguous(), W_k.contiguous(),
        core1.contiguous(), core2.contiguous(),
        O,
        T, D,
        R1, R2_padded,
        *q_flat.stride(), *k_flat.stride(), *v_flat.stride(),
        *W_q.stride(), *W_k.stride(),
        *core1.stride(), *core2.stride(),
        *O.stride(),
        H,
        CAUSAL=causal,
        BLOCK_R2=BLOCK_R2, BLOCK_D=BLOCK_D,
    )
    return O.reshape(B, H, T, D).to(q.dtype)


# -----------------------------------------------------------------------------
# Autograd Functions
# -----------------------------------------------------------------------------

class LsrOnlineFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, W_q, W_k, core, causal):
        out = _run_lsr_online(q.float(), k.float(), v.float(),
                              W_q.float(), W_k.float(), core.float(), causal)
        ctx.save_for_backward(q, k, v, W_q, W_k, core)
        ctx.causal = causal
        return out.to(q.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, W_q, W_k, core = ctx.saved_tensors
        causal = ctx.causal
        B, H, T, D = q.shape
        R = W_q.shape[-1]
        scale = 1.0 / math.sqrt(max(R, 1))

        BH = B * H
        head_idx = torch.arange(BH, device=q.device) % H

        W_q_per = W_q[head_idx]
        W_k_per = W_k[head_idx]
        core_per = core[head_idx]

        q_flat = q.reshape(BH, T, D)
        k_flat = k.reshape(BH, T, D)
        v_flat = v.reshape(BH, T, D)

        q_lr = torch.bmm(q_flat, W_q_per)
        k_lr = torch.bmm(k_flat, W_k_per)
        scores = scale * torch.bmm(q_lr * core_per[:, None, :], k_lr.transpose(1, 2))

        if causal:
            causal_mask = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(causal_mask, float("-inf"))
        P = torch.softmax(scores, dim=-1)

        go_flat = grad_output.reshape(BH, T, D)

        dv_flat = torch.bmm(P.transpose(1, 2), go_flat)
        dv = dv_flat.view(B, H, T, D).to(v.dtype)

        dP_flat = torch.bmm(go_flat, v_flat.transpose(1, 2))
        dp_sum = torch.sum(dP_flat * P, dim=-1, keepdim=True)
        dscores_flat = P * (dP_flat - dp_sum)
        if causal:
            dscores_flat = dscores_flat.masked_fill(causal_mask, 0.0)

        k_lr_scaled = k_lr * core_per[:, None, :]
        dq_lr_flat = scale * torch.bmm(dscores_flat, k_lr_scaled)

        q_lr_scaled = q_lr * core_per[:, None, :]
        dk_lr_flat = scale * torch.bmm(dscores_flat.transpose(1, 2), q_lr_scaled)

        tmp_core = torch.bmm(dscores_flat.transpose(1, 2), q_lr)
        dcore_flat = scale * torch.sum(tmp_core * k_lr, dim=1)
        dcore = dcore_flat.view(B, H, R).sum(dim=0)

        dq_flat = torch.bmm(dq_lr_flat, W_q_per.transpose(1, 2))
        dk_flat = torch.bmm(dk_lr_flat, W_k_per.transpose(1, 2))
        dq = dq_flat.view(B, H, T, D).to(q.dtype)
        dk = dk_flat.view(B, H, T, D).to(k.dtype)

        dW_q_updates = torch.bmm(q_flat.transpose(1, 2), dq_lr_flat)
        dW_k_updates = torch.bmm(k_flat.transpose(1, 2), dk_lr_flat)
        dW_q = dW_q_updates.view(B, H, D, R).sum(dim=0)
        dW_k = dW_k_updates.view(B, H, D, R).sum(dim=0)

        return dq, dk, dv, dW_q.to(W_q.dtype), dW_k.to(W_k.dtype), dcore.to(core.dtype), None


def _batched_kron(a, b):
    """Per-head Kronecker product: (H, R1), (H, R2) -> (H, R1*R2)."""
    H, R1 = a.shape
    _, R2 = b.shape
    return (a[:, :, None] * b[:, None, :]).reshape(H, R1 * R2)


class LsrKroneckerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, W_q, W_k, core1, core2, causal):
        out = _run_lsr_kronecker(
            q.float(), k.float(), v.float(),
            W_q.float(), W_k.float(), core1.float(), core2.float(), causal,
        )
        ctx.save_for_backward(q, k, v, W_q, W_k, core1, core2)
        ctx.causal = causal
        return out.to(q.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, W_q, W_k, core1, core2 = ctx.saved_tensors
        causal = ctx.causal
        B, H, T, D = q.shape
        R1 = core1.shape[-1]
        R2 = core2.shape[-1]
        R = R1 * R2
        scale = 1.0 / math.sqrt(max(R, 1))

        core_flat = _batched_kron(core1, core2)

        q_lr = torch.einsum("bhtd,hdr->bhtr", q, W_q)
        k_lr = torch.einsum("bhtd,hdr->bhtr", k, W_k)
        scores = torch.einsum("bhir,hr,bhjr->bhij", q_lr, core_flat, k_lr) * scale
        if causal:
            causal_mask = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(causal_mask, float("-inf"))
        P = torch.softmax(scores, dim=-1)

        grad_output = grad_output.reshape(B, H, T, D)

        dv = torch.einsum("bhij,bhid->bhjd", P, grad_output)
        dP = torch.einsum("bhid,bhjd->bhij", grad_output, v)
        dp_sum = torch.sum(dP * P, dim=-1, keepdim=True)
        dscores = P * (dP - dp_sum)
        if causal:
            dscores = dscores.masked_fill(causal_mask, 0.0)

        dq_lr = scale * torch.einsum("bhts,bhsr,hr->bhtr", dscores, k_lr, core_flat)
        dk_lr = scale * torch.einsum("bhts,bhtr,hr->bhsr", dscores, q_lr, core_flat)
        dcore_flat = scale * torch.einsum("bhts,bhtr,bhsr->hr", dscores, q_lr, k_lr)

        dq = torch.einsum("bhtr,hdr->bhtd", dq_lr, W_q)
        dk = torch.einsum("bhsr,hdr->bhsd", dk_lr, W_k)
        dW_q = torch.einsum("bhtd,bhtr->hdr", q, dq_lr)
        dW_k = torch.einsum("bhsd,bhsr->hdr", k, dk_lr)

        dcore1 = dcore_flat.view(H, R1, R2).sum(dim=2)
        dcore2 = dcore_flat.view(H, R1, R2).sum(dim=1)

        return dq, dk, dv, dW_q.to(W_q.dtype), dW_k.to(W_k.dtype), dcore1.to(core1.dtype), dcore2.to(core2.dtype), None


class LsrFusedFunction(torch.autograd.Function):
    """Autograd function for fused QKV+attention+output projection."""

    @staticmethod
    def forward(ctx, x, W_qkv, W_q_lsr, W_k_lsr, core, W_o, causal):
        global _LSR_FUSED_FALLBACK_WARNED
        try:
            out = _run_lsr_fused_qkvo(
                x.float(), W_qkv.float(),
                W_q_lsr.float(), W_k_lsr.float(), core.float(),
                W_o.float(), causal
            )
        except OutOfResources:
            if not _LSR_FUSED_FALLBACK_WARNED:
                print("[LSR] Fused kernel exceeded shared memory; falling back to unfused path.")
                _LSR_FUSED_FALLBACK_WARNED = True

            B, T, D_MODEL = x.shape
            H = W_q_lsr.shape[0]
            D_HEAD = W_q_lsr.shape[1]

            W_q = W_qkv[:, :D_MODEL]
            W_k = W_qkv[:, D_MODEL:2 * D_MODEL]
            W_v = W_qkv[:, 2 * D_MODEL:]

            q = (x @ W_q).view(B, T, H, D_HEAD).transpose(1, 2)
            k = (x @ W_k).view(B, T, H, D_HEAD).transpose(1, 2)
            v = (x @ W_v).view(B, T, H, D_HEAD).transpose(1, 2)

            R = W_q_lsr.shape[2]
            scale = 1.0 / math.sqrt(max(R, 1))
            q_lr = torch.einsum("bhtd,hdr->bhtr", q.float(), W_q_lsr.float()) * core.float()[None, :, None, :] * scale
            k_lr = torch.einsum("bhtd,hdr->bhtr", k.float(), W_k_lsr.float())
            scores = torch.einsum("bhir,bhjr->bhij", q_lr, k_lr)
            if causal:
                causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
                scores = scores.masked_fill(causal_mask, float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            attn_out = torch.einsum("bhij,bhjd->bhid", attn, v.float())
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D_MODEL)
            out = attn_out @ W_o.float()

        ctx.save_for_backward(x, W_qkv, W_q_lsr, W_k_lsr, core, W_o)
        ctx.causal = causal
        return out.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        # Fall back to PyTorch for backward pass (still fast due to fused forward)
        x, W_qkv, W_q_lsr, W_k_lsr, core, W_o = ctx.saved_tensors
        causal = ctx.causal
        B, T, D_MODEL = x.shape
        H = W_q_lsr.shape[0]
        D_HEAD = W_q_lsr.shape[1]
        R = W_q_lsr.shape[2]
        scale = 1.0 / math.sqrt(max(R, 1))

        # Recompute forward pass for gradients
        W_q = W_qkv[:, :D_MODEL]  # (D_MODEL, D_MODEL)
        W_k = W_qkv[:, D_MODEL:2*D_MODEL]
        W_v = W_qkv[:, 2*D_MODEL:]

        # Q, K, V projections
        Q_proj = x @ W_q  # (B, T, D_MODEL)
        K_proj = x @ W_k
        V_proj = x @ W_v

        # Reshape for multi-head
        Q = Q_proj.view(B, T, H, D_HEAD).transpose(1, 2)  # (B, H, T, D_HEAD)
        K = K_proj.view(B, T, H, D_HEAD).transpose(1, 2)
        V = V_proj.view(B, T, H, D_HEAD).transpose(1, 2)

        # LSR projections
        Q_lr = torch.einsum("bhtd,hdr->bhtr", Q, W_q_lsr) * core[None, :, None, :] * scale
        K_lr = torch.einsum("bhtd,hdr->bhtr", K, W_k_lsr)

        # Attention scores
        scores = torch.einsum("bhir,bhjr->bhij", Q_lr, K_lr)
        if causal:
            causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(causal_mask, float("-inf"))
        P = torch.softmax(scores, dim=-1)

        # Attention output
        attn_out = torch.einsum("bhij,bhjd->bhid", P, V)  # (B, H, T, D_HEAD)
        attn_out_reshaped = attn_out.transpose(1, 2).contiguous().view(B, T, D_MODEL)

        # Backward through output projection
        grad_output = grad_output.to(x.dtype)
        dW_o = attn_out_reshaped.transpose(1, 2).reshape(D_MODEL, -1) @ grad_output.reshape(-1, D_MODEL) / B
        d_attn_out = grad_output @ W_o.T
        d_attn_out = d_attn_out.view(B, T, H, D_HEAD).transpose(1, 2)

        # Backward through attention
        dV = torch.einsum("bhij,bhid->bhjd", P, d_attn_out)
        dP = torch.einsum("bhid,bhjd->bhij", d_attn_out, V)
        dp_sum = torch.sum(dP * P, dim=-1, keepdim=True)
        dscores = P * (dP - dp_sum)
        if causal:
            dscores = dscores.masked_fill(causal_mask, 0.0)

        # Backward through LSR
        dQ_lr = torch.einsum("bhij,bhjr->bhir", dscores, K_lr) * scale
        dK_lr = torch.einsum("bhij,bhir->bhjr", dscores, Q_lr * core[None, :, None, :]) * scale
        dcore = torch.einsum("bhij,bhir,bhjr->hr", dscores, Q_lr / (core[None, :, None, :] + 1e-8) * core[None, :, None, :], K_lr) * scale
        dcore = dcore.sum(dim=0) if dcore.dim() > 2 else dcore

        dW_q_lsr = torch.einsum("bhtd,bhtr->hdr", Q, dQ_lr * core[None, :, None, :])
        dW_k_lsr = torch.einsum("bhtd,bhtr->hdr", K, dK_lr)

        dQ = torch.einsum("bhtr,hdr->bhtd", dQ_lr * core[None, :, None, :], W_q_lsr)
        dK = torch.einsum("bhtr,hdr->bhtd", dK_lr, W_k_lsr)

        # Backward through QKV projections
        dQ_proj = dQ.transpose(1, 2).contiguous().view(B, T, D_MODEL)
        dK_proj = dK.transpose(1, 2).contiguous().view(B, T, D_MODEL)
        dV_proj = dV.transpose(1, 2).contiguous().view(B, T, D_MODEL)

        dx = dQ_proj @ W_q.T + dK_proj @ W_k.T + dV_proj @ W_v.T

        dW_q = x.transpose(1, 2).reshape(D_MODEL, -1) @ dQ_proj.reshape(-1, D_MODEL) / B
        dW_k = x.transpose(1, 2).reshape(D_MODEL, -1) @ dK_proj.reshape(-1, D_MODEL) / B
        dW_v = x.transpose(1, 2).reshape(D_MODEL, -1) @ dV_proj.reshape(-1, D_MODEL) / B

        dW_qkv = torch.cat([dW_q, dW_k, dW_v], dim=1)

        return dx.to(x.dtype), dW_qkv.to(W_qkv.dtype), dW_q_lsr.to(W_q_lsr.dtype), dW_k_lsr.to(W_k_lsr.dtype), dcore.to(core.dtype), dW_o.to(W_o.dtype), None


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def lsr_attention_online(q, k, v, W_q, W_k, core, causal=True):
    """LSR attention with online softmax."""
    return LsrOnlineFunction.apply(q, k, v, W_q, W_k, core, causal)


def lsr_attention_kronecker(q, k, v, W_q, W_k, core1, core2, causal=True):
    """LSR attention with Kronecker factorized core."""
    return LsrKroneckerFunction.apply(q, k, v, W_q, W_k, core1, core2, causal)


def lsr_attention_fused(x, W_qkv, W_q_lsr, W_k_lsr, core, W_o, causal=True):
    """Fused QKV projection + LSR attention + output projection."""
    return LsrFusedFunction.apply(x, W_qkv, W_q_lsr, W_k_lsr, core, W_o, causal)


# -----------------------------------------------------------------------------
# Module
# -----------------------------------------------------------------------------

class MultiHeadLSRAttention(nn.Module):
    """Multi-head LSR attention with Triton kernels."""

    def __init__(self, d_model, num_heads, lsr_rank=32, use_kronecker=False, r1=2):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.lsr_rank = lsr_rank
        self.use_kronecker = use_kronecker

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        if use_kronecker:
            assert lsr_rank % r1 == 0
            r2 = lsr_rank // r1
            self.W_q_lsr = nn.Parameter(torch.randn(num_heads, self.d_head, lsr_rank) / math.sqrt(self.d_head))
            self.W_k_lsr = nn.Parameter(torch.randn(num_heads, self.d_head, lsr_rank) / math.sqrt(self.d_head))
            self.lsr_core1 = nn.Parameter(torch.ones(num_heads, r1))
            self.lsr_core2 = nn.Parameter(torch.ones(num_heads, r2))
        else:
            self.W_q_lsr = nn.Parameter(torch.randn(num_heads, self.d_head, lsr_rank) / math.sqrt(self.d_head))
            self.W_k_lsr = nn.Parameter(torch.randn(num_heads, self.d_head, lsr_rank) / math.sqrt(self.d_head))
            self.lsr_core = nn.Parameter(torch.ones(num_heads, lsr_rank))

    def forward(self, x):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)

        if self.use_kronecker:
            y = lsr_attention_kronecker(q, k, v, self.W_q_lsr, self.W_k_lsr, self.lsr_core1, self.lsr_core2, causal=True)
        else:
            y = lsr_attention_online(q, k, v, self.W_q_lsr, self.W_k_lsr, self.lsr_core, causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.o_proj(y)


class MultiHeadLSRAttentionFused(nn.Module):
    """Multi-head LSR attention with fused QKV+attention+output projection.

    This module fuses all linear projections (Q, K, V, O) with the attention
    computation into a single Triton kernel, eliminating memory round-trips
    and kernel launch overhead.
    """

    def __init__(self, d_model, num_heads, lsr_rank=32):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.lsr_rank = lsr_rank

        # Fused QKV projection: (d_model, 3*d_model)
        # Layout: [W_q | W_k | W_v] where each is (d_model, d_model)
        self.W_qkv = nn.Parameter(torch.randn(d_model, 3 * d_model) / math.sqrt(d_model))

        # Output projection: (d_model, d_model)
        self.W_o = nn.Parameter(torch.randn(d_model, d_model) / math.sqrt(d_model))

        # LSR projections
        self.W_q_lsr = nn.Parameter(torch.randn(num_heads, self.d_head, lsr_rank) / math.sqrt(self.d_head))
        self.W_k_lsr = nn.Parameter(torch.randn(num_heads, self.d_head, lsr_rank) / math.sqrt(self.d_head))
        self.lsr_core = nn.Parameter(torch.ones(num_heads, lsr_rank))

    def forward(self, x):
        """Forward pass with fused kernel.

        Args:
            x: Input tensor (B, T, d_model)

        Returns:
            Output tensor (B, T, d_model)
        """
        return lsr_attention_fused(
            x, self.W_qkv, self.W_q_lsr, self.W_k_lsr, self.lsr_core, self.W_o, causal=True
        )
