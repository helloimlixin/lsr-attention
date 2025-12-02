"""Triton fused Low Separation Rank (LSR) attention."""

import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


# -----------------------------------------------------------------------------
# Kernels
# -----------------------------------------------------------------------------

@triton.jit
def lsr_fused_kernel(
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
):
    """Fused forward kernel for a flat core of rank R."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_r = tl.arange(0, BLOCK_R)

    m_mask = offs_m < T
    d_mask = offs_d < D
    r_mask = offs_r < ORIGINAL_R

    wq_ptrs = Wq_ptr + h * stride_wq_h + offs_d[:, None] * stride_wq_d + offs_r[None, :] * stride_wq_r
    wk_ptrs = Wk_ptr + h * stride_wk_h + offs_d[:, None] * stride_wk_d + offs_r[None, :] * stride_wk_r
    Wq_block = tl.load(wq_ptrs, mask=d_mask[:, None] & r_mask[None, :], other=0.0)
    Wk_block = tl.load(wk_ptrs, mask=d_mask[:, None] & r_mask[None, :], other=0.0)

    core_ptrs = Core_ptr + h * stride_core_h + offs_r * stride_core_r
    core_block = tl.load(core_ptrs, mask=r_mask, other=0.0)

    q_ptrs = Q_ptr + pid_bh * stride_q_bh + offs_m[:, None] * stride_q_t + offs_d[None, :] * stride_q_d
    Q_block = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)
    Q_lr = tl.dot(Q_block, Wq_block)

    scale = 1.0 / tl.sqrt(tl.cast(ORIGINAL_R, tl.float32))
    Q_lr = Q_lr * core_block[None, :] * scale

    if CAUSAL:
        n_blocks = tl.minimum(tl.cdiv((pid_m + 1) * BLOCK_M, BLOCK_N), tl.cdiv(T, BLOCK_N))
    else:
        n_blocks = tl.cdiv(T, BLOCK_N)

    row_max = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)

    for n_block in range(n_blocks):
        offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < T

        k_ptrs = K_ptr + pid_bh * stride_k_bh + offs_n[:, None] * stride_k_t + offs_d[None, :] * stride_k_d
        K_block = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
        K_lr = tl.dot(K_block, Wk_block)

        scores = tl.dot(Q_lr, tl.trans(K_lr))

        if CAUSAL:
            causal_mask = offs_n[None, :] <= offs_m[:, None]
            scores = tl.where(causal_mask, scores, float("-inf"))
        scores = tl.where(n_mask[None, :], scores, float("-inf"))

        block_max = tl.max(scores, axis=1)
        row_max = tl.maximum(row_max, block_max)

    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for n_block in range(n_blocks):
        offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < T

        k_ptrs = K_ptr + pid_bh * stride_k_bh + offs_n[:, None] * stride_k_t + offs_d[None, :] * stride_k_d
        K_block = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
        K_lr = tl.dot(K_block, Wk_block)

        scores = tl.dot(Q_lr, tl.trans(K_lr))

        if CAUSAL:
            causal_mask = offs_n[None, :] <= offs_m[:, None]
            scores = tl.where(causal_mask, scores, float("-inf"))
        scores = tl.where(n_mask[None, :], scores, float("-inf"))

        exp_scores = tl.exp(scores - row_max[:, None])
        row_sum += tl.sum(exp_scores, axis=1)

        v_ptrs = V_ptr + pid_bh * stride_v_bh + offs_n[:, None] * stride_v_t + offs_d[None, :] * stride_v_d
        V_block = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
        acc += tl.dot(exp_scores.to(V_block.dtype), V_block)

    acc = acc / row_sum[:, None]

    o_ptrs = O_ptr + pid_bh * stride_o_bh + offs_m[:, None] * stride_o_t + offs_d[None, :] * stride_o_d
    tl.store(o_ptrs, acc, mask=m_mask[:, None] & d_mask[None, :])


@triton.jit
def lsr_fused_kernel_factorized(
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
    """Fused forward kernel for a two-factor Kronecker core core1⊗core2."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_r2 = tl.arange(0, BLOCK_R2)

    m_mask = offs_m < T
    d_mask = offs_d < D
    r2_mask = offs_r2 < R2

    core1_ptrs = Core1_ptr + h * stride_core1_h + tl.arange(0, R1) * stride_core1_r
    core1 = tl.load(core1_ptrs)
    core2_ptrs = Core2_ptr + h * stride_core2_h + offs_r2 * stride_core2_r
    core2 = tl.load(core2_ptrs, mask=r2_mask, other=0.0)

    q_ptrs = Q_ptr + pid_bh * stride_q_bh + offs_m[:, None] * stride_q_t + offs_d[None, :] * stride_q_d
    Q_block = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)

    if CAUSAL:
        n_blocks = tl.minimum(tl.cdiv((pid_m + 1) * BLOCK_M, BLOCK_N), tl.cdiv(T, BLOCK_N))
    else:
        n_blocks = tl.cdiv(T, BLOCK_N)

    row_max = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)

    for n_block in range(n_blocks):
        offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < T

        k_ptrs = K_ptr + pid_bh * stride_k_bh + offs_n[:, None] * stride_k_t + offs_d[None, :] * stride_k_d
        K_block = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)

        scores = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for r1 in range(R1):
            wq_ptrs_r1 = Wq_ptr + h * stride_wq_h + offs_d[:, None] * stride_wq_d + (r1 * BLOCK_R2 + offs_r2[None, :]) * stride_wq_r
            wk_ptrs_r1 = Wk_ptr + h * stride_wk_h + offs_d[:, None] * stride_wk_d + (r1 * BLOCK_R2 + offs_r2[None, :]) * stride_wk_r
            Wq_block = tl.load(wq_ptrs_r1, mask=d_mask[:, None] & r2_mask[None, :], other=0.0)
            Wk_block = tl.load(wk_ptrs_r1, mask=d_mask[:, None] & r2_mask[None, :], other=0.0)

            q_lr = tl.dot(Q_block, Wq_block)
            k_lr = tl.dot(K_block, Wk_block)
            partial = tl.dot(q_lr * core2[None, :], tl.trans(k_lr))
            scores += core1[r1] * partial

        scale = 1.0 / tl.sqrt(tl.float32(R1 * R2))
        scores = scores * scale

        if CAUSAL:
            causal_mask = offs_n[None, :] <= offs_m[:, None]
            scores = tl.where(causal_mask, scores, float("-inf"))
        scores = tl.where(n_mask[None, :], scores, float("-inf"))

        block_max = tl.max(scores, axis=1)
        row_max = tl.maximum(row_max, block_max)

    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for n_block in range(n_blocks):
        offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < T

        k_ptrs = K_ptr + pid_bh * stride_k_bh + offs_n[:, None] * stride_k_t + offs_d[None, :] * stride_k_d
        K_block = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)

        scores = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for r1 in range(R1):
            wq_ptrs_r1 = Wq_ptr + h * stride_wq_h + offs_d[:, None] * stride_wq_d + (r1 * BLOCK_R2 + offs_r2[None, :]) * stride_wq_r
            wk_ptrs_r1 = Wk_ptr + h * stride_wk_h + offs_d[:, None] * stride_wk_d + (r1 * BLOCK_R2 + offs_r2[None, :]) * stride_wk_r
            Wq_block = tl.load(wq_ptrs_r1, mask=d_mask[:, None] & r2_mask[None, :], other=0.0)
            Wk_block = tl.load(wk_ptrs_r1, mask=d_mask[:, None] & r2_mask[None, :], other=0.0)

            q_lr = tl.dot(Q_block, Wq_block)
            k_lr = tl.dot(K_block, Wk_block)
            partial = tl.dot(q_lr * core2[None, :], tl.trans(k_lr))
            scores += core1[r1] * partial

        scale = 1.0 / tl.sqrt(tl.float32(R1 * R2))
        scores = scores * scale

        if CAUSAL:
            causal_mask = offs_n[None, :] <= offs_m[:, None]
            scores = tl.where(causal_mask, scores, float("-inf"))
        scores = tl.where(n_mask[None, :], scores, float("-inf"))

        exp_scores = tl.exp(scores - row_max[:, None])
        row_sum += tl.sum(exp_scores, axis=1)

        v_ptrs = V_ptr + pid_bh * stride_v_bh + offs_n[:, None] * stride_v_t + offs_d[None, :] * stride_v_d
        V_block = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
        acc += tl.dot(exp_scores.to(V_block.dtype), V_block)

    acc = acc / row_sum[:, None]
    o_ptrs = O_ptr + pid_bh * stride_o_bh + offs_m[:, None] * stride_o_t + offs_d[None, :] * stride_o_d
    tl.store(o_ptrs, acc, mask=m_mask[:, None] & d_mask[None, :])


# -----------------------------------------------------------------------------
# Python wrappers
# -----------------------------------------------------------------------------

def _run_fused(q, k, v, W_q, W_k, core, causal=True):
    B, H, T, D = q.shape
    R = W_q.shape[-1]
    BH = B * H

    R_padded = max(R, 16)
    if R_padded != R:
        pad = R_padded - R
        W_q = torch.nn.functional.pad(W_q, (0, pad), value=0.0)
        W_k = torch.nn.functional.pad(W_k, (0, pad), value=0.0)
        core = torch.nn.functional.pad(core, (0, pad), value=0.0)

    q_flat = q.reshape(BH, T, D).contiguous()
    k_flat = k.reshape(BH, T, D).contiguous()
    v_flat = v.reshape(BH, T, D).contiguous()

    O = torch.empty(BH, T, D, device=q.device, dtype=torch.float32)

    BLOCK_M = min(64, T)
    BLOCK_N = min(64, T)
    BLOCK_R = min(32, R_padded)
    BLOCK_D = min(64, D)

    grid = (triton.cdiv(T, BLOCK_M), BH)
    lsr_fused_kernel[grid](
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
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_R=BLOCK_R, BLOCK_D=BLOCK_D,
        ORIGINAL_R=R,
    )
    return O.reshape(B, H, T, D).to(q.dtype)


def _run_fused_factorized(q, k, v, W_q, W_k, core1, core2, causal=True):
    B, H, T, D = q.shape
    R1 = core1.shape[-1]
    R2 = core2.shape[-1]
    BH = B * H

    BLOCK_R2 = min(32, triton.next_power_of_2(R2))
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

    BLOCK_M = min(64, T)
    BLOCK_N = min(64, T)
    BLOCK_D = min(64, D)

    grid = (triton.cdiv(T, BLOCK_M), BH)
    lsr_fused_kernel_factorized[grid](
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
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_R2=BLOCK_R2, BLOCK_D=BLOCK_D,
    )
    return O.reshape(B, H, T, D).to(q.dtype)


# -----------------------------------------------------------------------------
# Autograd
# -----------------------------------------------------------------------------

class LsrTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, W_q, W_k, core, causal):
        out = _run_fused(q.float(), k.float(), v.float(), W_q.float(), W_k.float(), core.float(), causal)
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


class LsrTritonFactorFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, W_q, W_k, core1, core2, causal):
        out = _run_fused_factorized(
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

        core_flat = torch.kron(core1, core2)  # (H, R)

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


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def lsr_attention_triton(q, k, v, W_q, W_k, core, causal=True):
    """LSR attention via fused Triton kernel with autograd."""
    return LsrTritonFunction.apply(q, k, v, W_q, W_k, core, causal)


def lsr_attention_triton_factorized(q, k, v, W_q, W_k, core1, core2, causal=True):
    """LSR attention with factorized core via fused Triton kernel."""
    return LsrTritonFactorFunction.apply(q, k, v, W_q, W_k, core1, core2, causal)

# Backward-compatible alias
lsr_attention_triton_fused = lsr_attention_triton


class MultiHeadLSRAttentionTriton(nn.Module):
    """Multi-head LSR attention using fused Triton kernels."""

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
        self.lsr_core = nn.Parameter(torch.ones(num_heads, lsr_rank))

    def forward(self, x):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)

        y = lsr_attention_triton(q, k, v, self.W_q_lsr, self.W_k_lsr, self.lsr_core, causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.o_proj(y)
