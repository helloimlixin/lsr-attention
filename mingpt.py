
"""
Minimal GPT implementation with an optional Kronecker-factored linear path.
All LSR-related code paths have been removed.
"""

# Ensure extensions/ is in sys.path for CUDA extension import
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "extensions"))

import math
import argparse
import time
from dataclasses import dataclass, replace
from typing import Optional, Tuple

try:
    from accelerate import Accelerator
except Exception:  # noqa: BLE001
    Accelerator = None

try:
    from tqdm.auto import tqdm
except Exception:  # noqa: BLE001
    tqdm = None

import torch
import torch.nn as nn
import torch.nn.functional as F
from data import make_batch, load_shakespeare

# Optional Triton for inference-only matmul
# Optional Triton for inference-only matmul
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:  # noqa: BLE001
    _HAS_TRITON = False

# Optional fused CUDA extension (will be a separate build)
try:
    import kron_fused  # type: ignore

    _HAS_KRON_FUSED = True
except Exception:  # noqa: BLE001
    kron_fused = None
    _HAS_KRON_FUSED = False


if _HAS_TRITON:
    @triton.jit
    def _matmul_kernel(
        A_ptr, B_ptr, C_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        k_iter = tl.cdiv(K, BLOCK_K)
        for i in range(0, k_iter):
            k_start = i * BLOCK_K
            k_idx = k_start + offs_k
            a_mask = (offs_m[:, None] < M) & (k_idx[None, :] < K)
            b_mask = (k_idx[:, None] < K) & (offs_n[None, :] < N)
            A = tl.load(A_ptr + offs_m[:, None] * stride_am + k_idx[None, :] * stride_ak, mask=a_mask, other=0.0)
            B = tl.load(B_ptr + k_idx[:, None] * stride_bk + offs_n[None, :] * stride_bn, mask=b_mask, other=0.0)
            acc += tl.dot(A, B)

        acc = acc.to(tl.float16)
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc, mask=c_mask)


    def triton_matmul_fixed(a: torch.Tensor, b: torch.Tensor, block_m: int = 128, block_n: int = 128, block_k: int = 32) -> torch.Tensor:
        # a: (M, K), b: (N, K) transposed; we want a @ b.T
        assert a.is_cuda and b.is_cuda, "Triton matmul requires CUDA tensors"
        M, K = a.shape
        N = b.shape[0]
        # Strides assume row-major for a, row-major for b (b is already transposed weight)
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)

        grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
        _matmul_kernel[grid](
            a, b,
            c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(1), b.stride(0),
            c.stride(0), c.stride(1),
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        )
        return c


class KronTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x2d: torch.Tensor, W: torch.Tensor, bias: Optional[torch.Tensor]):
        # x2d: (B*, in_features); W: (out_features, in_features)
        out = triton_matmul_fixed(x2d, W)
        ctx.save_for_backward(x2d, W, bias)
        return out if bias is None else out + bias

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x2d, W, bias = ctx.saved_tensors
        grad_x = grad_w = grad_b = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_out @ W
        if ctx.needs_input_grad[1]:
            grad_w = grad_out.t() @ x2d
        if bias is not None and ctx.needs_input_grad[2]:
            grad_b = grad_out.sum(dim=0)
        return grad_x, grad_w, grad_b


class KronCudaFusedFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x2d: torch.Tensor, A: torch.Tensor, B: torch.Tensor, bias: Optional[torch.Tensor], heads: int, head_dim: int, in_mult: int, out_mult: int):
        if not _HAS_KRON_FUSED:
            raise RuntimeError("kron_fused extension not available")
        # Expect x2d: (B*, in_features), A: (rank, out_mult*heads, in_mult*heads), B: (rank, head_dim, head_dim)
        y = kron_fused.forward(x2d, A, B, heads, head_dim, in_mult, out_mult)
        ctx.save_for_backward(x2d, A, B, bias)
        ctx.meta = (heads, head_dim, in_mult, out_mult)
        return y if bias is None else y + bias

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        if not _HAS_KRON_FUSED:
            raise RuntimeError("kron_fused extension not available")
        x2d, A, B, bias = ctx.saved_tensors
        heads, head_dim, in_mult, out_mult = ctx.meta
        grad_x, grad_A, grad_B = kron_fused.backward(grad_out.contiguous(), x2d, A, B, heads, head_dim, in_mult, out_mult)
        grad_bias = grad_out.sum(dim=0) if (bias is not None and ctx.needs_input_grad[3]) else None
        return grad_x, grad_A, grad_B, grad_bias, None, None, None, None


@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 50257
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = False
    use_kron: bool = False
    kron_rank: int = 1
    kron_infer_dense: bool = False
    kron_fused: bool = False
    kron_triton_infer: bool = False
    kron_triton_autograd: bool = False
    kron_cuda_fused: bool = False


class KroneckerLinear(nn.Module):
    r"""Linear layer using a small Kronecker factorization.

    The weight matrix W is represented as the sum of rank Kronecker products:
    W = \sum_r (A_r \otimes B_r). We choose factor shapes to map between
    feature dimensions built from (n_head, head_dim) pairs, which keeps each
    factor small and CUDA-friendly.
    """

    def __init__(
        self,
        heads: int,
        head_dim: int,
        in_mult: int = 1,
        out_mult: int = 1,
        rank: int = 1,
        bias: bool = False,
        infer_dense: bool = False,
        fused: bool = False,
        triton_infer: bool = False,
        triton_autograd: bool = False,
        cuda_fused: bool = False,
    ):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.in_mult = in_mult
        self.out_mult = out_mult
        self.rank = rank
        self.infer_dense = infer_dense
        self.fused = fused
        self.triton_infer = triton_infer and _HAS_TRITON
        self.triton_autograd = triton_autograd and _HAS_TRITON
        self.cuda_fused = cuda_fused and _HAS_KRON_FUSED

        self.in_features = heads * head_dim * in_mult
        self.out_features = heads * head_dim * out_mult

        self.A = nn.Parameter(torch.empty(rank, out_mult * heads, in_mult * heads))
        self.B = nn.Parameter(torch.empty(rank, head_dim, head_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()
        self._cached_dense = None
        self._cached_device = None
        self._cached_dtype = None

    def reset_parameters(self):
        for r in range(self.rank):
            nn.init.normal_(self.A[r], mean=0.0, std=0.02)
            nn.init.normal_(self.B[r], mean=0.0, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *prefix, feat = x.shape
        assert feat == self.in_features, f"Expected last dim {self.in_features}, got {feat}"

        # Use explicit batched matmul for speed
        B = x.shape[0] if x.ndim == 2 else x.shape[0] * x.shape[1]
        hd = self.head_dim
        R = self.rank
        In = self.in_mult * self.heads
        Out = self.out_mult * self.heads

        x_flat = x.reshape(-1, hd, In)  # (B, Dh, In)
        # First contraction: (B, Dh, In) x (R, In, Out) -> (B, R, Dh, Out)
        # We do this by expanding x and using bmm
        x_exp = x_flat.unsqueeze(1).expand(-1, R, -1, -1).reshape(-1, hd, In)  # (B*R, Dh, In)
        A_exp = self.A.reshape(R, Out, In).expand(B, -1, -1, -1).reshape(-1, Out, In)  # (B*R, Out, In)
        mid = torch.bmm(x_exp, A_exp.transpose(1, 2))  # (B*R, Dh, Out)
        mid = mid.view(B, R, hd, Out)  # (B, R, Dh, Out)

        # Second contraction: (B, R, Dh, Out) x (R, Dh, Dh) -> (B, R, Dh, Out)
        # For each rank, do bmm over Dh
        B_exp = self.B.unsqueeze(0).expand(B, -1, -1, -1).reshape(-1, hd, hd)  # (B*R, Dh, Dh)
        mid2 = mid.permute(0,1,3,2).reshape(-1, Out, hd)  # (B*R, Out, Dh)
        y = torch.bmm(mid2, B_exp.transpose(1,2))  # (B*R, Out, Dh)
        y = y.view(B, R, Out, hd).permute(0,1,3,2)  # (B, R, Dh, Out)

        # Sum over rank
        acc = y.sum(dim=1)  # (B, Dh, Out)
        out = acc.transpose(1, 2).contiguous().view(-1, self.out_features)
        if self.bias is not None:
            out = out + self.bias
        return out.view(*prefix, self.out_features)

    @torch.no_grad()
    def materialize_dense(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self._cached_dense is not None and self._cached_device == device and self._cached_dtype == dtype:
            return self._cached_dense

        # Build dense weight: sum_r block-wise Hadamard product matching einsum path
        weight = torch.zeros(self.out_features, self.in_features, device=device, dtype=dtype)
        hd = self.head_dim
        OM = self.out_mult * self.heads
        IM = self.in_mult * self.heads
        for r in range(self.rank):
            Ar = self.A[r].to(device=device, dtype=dtype)
            Br = self.B[r].to(device=device, dtype=dtype)
            # Block-wise Hadamard: W[i*hd:(i+1)*hd, j*hd:(j+1)*hd] = Br * Ar[i, j]
            for i in range(OM):
                for j in range(IM):
                    weight[i*hd:(i+1)*hd, j*hd:(j+1)*hd] += Br * Ar[i, j]

        self._cached_dense = weight
        self._cached_device = device
        self._cached_dtype = dtype
        return weight

    def test_materialize_dense(self):
        # Test that materialize_dense matches einsum path for random input
        x = torch.randn(2, self.in_features, device=self.A.device, dtype=self.A.dtype)
        # Einsum path
        x_flat = x.reshape(-1, self.head_dim, self.in_mult * self.heads)
        mid = torch.einsum("bdi,roi->bdro", x_flat, self.A)
        y = torch.einsum("rde,bdro->bero", self.B, mid)
        acc = y.sum(dim=2)
        out_einsum = acc.transpose(1, 2).contiguous().view(-1, self.out_features)
        # Dense path
        W = self.materialize_dense(self.A.device, self.A.dtype)
        out_dense = torch.matmul(x, W.t())
        # Compare
        max_diff = (out_einsum - out_dense).abs().max().item()
        print(f"[Debug] KroneckerLinear materialize_dense max diff: {max_diff}")
        return max_diff


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        if config.use_kron:
            self.c_attn = KroneckerLinear(
                config.n_head,
                self.head_dim,
                out_mult=3,
                in_mult=1,
                rank=config.kron_rank,
                bias=config.bias,
                infer_dense=config.kron_infer_dense,
                fused=config.kron_fused,
                triton_infer=config.kron_triton_infer,
                cuda_fused=config.kron_cuda_fused,
            )
            self.c_proj = KroneckerLinear(
                config.n_head,
                self.head_dim,
                out_mult=1,
                in_mult=1,
                rank=config.kron_rank,
                bias=config.bias,
                infer_dense=config.kron_infer_dense,
                fused=config.kron_fused,
                triton_infer=config.kron_triton_infer,
                cuda_fused=config.kron_cuda_fused,
            )
        else:
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
            self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("bias", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        if config.use_kron:
            self.c_fc = KroneckerLinear(
                config.n_head,
                config.n_embd // config.n_head,
                out_mult=4,
                in_mult=1,
                rank=config.kron_rank,
                bias=config.bias,
                infer_dense=config.kron_infer_dense,
                fused=config.kron_fused,
                triton_infer=config.kron_triton_infer,
                cuda_fused=config.kron_cuda_fused,
            )
            self.c_proj = KroneckerLinear(
                config.n_head,
                config.n_embd // config.n_head,
                out_mult=1,
                in_mult=4,
                rank=config.kron_rank,
                bias=config.bias,
                infer_dense=config.kron_infer_dense,
                fused=config.kron_fused,
                triton_infer=config.kron_triton_infer,
                cuda_fused=config.kron_cuda_fused,
            )
        else:
            self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
            self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "wpe": nn.Embedding(config.block_size, config.n_embd),
            "drop": nn.Dropout(config.dropout),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            "ln_f": nn.LayerNorm(config.n_embd, bias=config.bias),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.transformer["wte"].weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, KroneckerLinear):
            module.reset_parameters()
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.size()
        assert T <= self.config.block_size, "Sequence length exceeds block size"

        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)

        tok_emb = self.transformer["wte"](idx)
        pos_emb = self.transformer["wpe"](pos)
        x = self.transformer["drop"](tok_emb + pos_emb)
        for block in self.transformer["h"]:
            x = block(x)
        x = self.transformer["ln_f"](x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = torch.where(logits < v[:, [-1]], torch.full_like(logits, -float('inf')), logits)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def get_batch(block_size: int, device: torch.device, batch_size: int, tokens: Optional[torch.Tensor] = None):
    if tokens is None:
        data = torch.randint(0, 1000, (10000,), device=device)
        ix = torch.randint(len(data) - block_size, (batch_size,), device=device)
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
        return x, y
    return make_batch(tokens, batch_size, block_size, device)


def train(config: GPTConfig, device: torch.device, max_iters: int = 50, eval_interval: int = 10, batch_size: int = 32, tokens: Optional[torch.Tensor] = None):
    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)

    model.train()
    for step in range(max_iters):
        xb, yb = get_batch(config.block_size, device, batch_size, tokens)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(xb, yb)
        loss.backward()
        optimizer.step()
        if step % eval_interval == 0:
            print(f"step {step}: loss {loss.item():.4f}")


def _sync_if_cuda(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def benchmark_variant(config: GPTConfig, device: torch.device, batch_size: int, steps: int, gen_tokens: int, tokens: Optional[torch.Tensor] = None):
    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)

    train_times = []
    last_loss = 0.0
    model.train()
    for _ in range(steps):
        xb, yb = get_batch(config.block_size, device, batch_size, tokens)
        optimizer.zero_grad(set_to_none=True)
        t0 = time.perf_counter()
        _, loss = model(xb, yb)
        loss.backward()
        optimizer.step()
        _sync_if_cuda(device)
        train_times.append(time.perf_counter() - t0)
        last_loss = loss.item()

    avg_train = sum(train_times) / len(train_times)
    tokens_per_step = batch_size * config.block_size
    train_tps = tokens_per_step / avg_train

    model.eval()
    # Warm materialized dense weights for inference path to avoid first-call penalty.
    param_dtype = next(model.parameters()).dtype
    for m in model.modules():
        if isinstance(m, KroneckerLinear) and m.infer_dense:
            m.materialize_dense(device, param_dtype)

    with torch.no_grad():
        xb, _ = get_batch(config.block_size, device, batch_size, tokens)
        t1 = time.perf_counter()
        _ = model(xb)
        _sync_if_cuda(device)
        fwd_time = time.perf_counter() - t1

        t2 = time.perf_counter()
        _ = model.generate(xb[:1], max_new_tokens=gen_tokens)
        _sync_if_cuda(device)
        gen_time = time.perf_counter() - t2

    return {
        "train_ms": avg_train * 1000.0,
        "train_tps": train_tps,
        "loss": last_loss,
        "fwd_ms": fwd_time * 1000.0,
        "gen_ms": gen_time * 1000.0,
    }


def _decode_tokens(tokenizer, tokens: torch.Tensor) -> str:
    if tokenizer is None:
        return "[no tokenizer available]"
    return tokenizer.decode(tokens.tolist(), skip_special_tokens=True)


def accelerate_train(
    config: GPTConfig,
    batch_size: int,
    steps: int,
    dataset: str,
    sample_after: bool,
    sample_tokens: int,
    sample_prompt: str,
    text_path: Optional[str] = None,
):
    if config.use_kron:
        print("[Accelerate] Using Kronecker model (kron_*)")
    else:
        print("[Accelerate] Using baseline model (standard Linear)")
    # ...existing code continues...

    accelerator = Accelerator()
    device = accelerator.device

    tokens = None
    tokenizer = None
    if dataset == "shakespeare":
        tokenizer, train_ids, _ = load_shakespeare(text_path=text_path)
        tokens = train_ids.to(device)

    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    model, optimizer = accelerator.prepare(model, optimizer)

    train_times = []
    last_loss = 0.0

    model.train()
    pbar = tqdm(range(steps), disable=not accelerator.is_local_main_process)
    for _ in pbar:
        xb, yb = get_batch(config.block_size, device, batch_size, tokens)
        optimizer.zero_grad(set_to_none=True)
        t0 = time.perf_counter()
        _, loss = model(xb, yb)
        accelerator.backward(loss)
        optimizer.step()
        accelerator.wait_for_everyone()
        t1 = time.perf_counter()
        train_times.append(t1 - t0)
        last_loss = loss.item()
        if accelerator.is_local_main_process:
            pbar.set_postfix({"loss": f"{last_loss:.4f}"})
    accelerator.wait_for_everyone()

    # Compute throughput on main process
    if accelerator.is_local_main_process:
        avg_train = sum(train_times) / len(train_times)
        tokens_per_step = batch_size * config.block_size
        train_tps = tokens_per_step / avg_train
        print(f"[Accelerate] train avg: {avg_train*1000:.2f} ms/step | tokens/s: {train_tps:.1f} | loss: {last_loss:.4f}")

    if sample_after:
        if accelerator.is_local_main_process:
            base_model = accelerator.unwrap_model(model)
            base_model.eval()
            with torch.no_grad():
                if tokenizer is not None:
                    prompt_ids = tokenizer(sample_prompt, return_tensors="pt").input_ids.to(device)
                else:
                    prompt_ids = torch.randint(0, config.vocab_size, (1, 8), device=device)

                # Debug: print prompt token ids and decoded prompt
                print("[Debug] Prompt token ids:", prompt_ids[0].tolist())
                if tokenizer is not None:
                    decoded_prompt = tokenizer.decode(prompt_ids[0])
                    print("[Debug] Decoded prompt:", decoded_prompt)
                # Forward latency (single batch)
                t_fwd0 = time.perf_counter()
                _ = base_model(prompt_ids)
                t_fwd1 = time.perf_counter()

                # Generation timing with tqdm progress bar
                t_gen0 = time.perf_counter()
                gen_ids = prompt_ids.clone()
                for _ in tqdm(range(sample_tokens), desc="Generating", ncols=80):
                    gen_ids = base_model.generate(gen_ids, max_new_tokens=1)
                t_gen1 = time.perf_counter()

                fwd_ms = (t_fwd1 - t_fwd0) * 1000.0
                gen_ms = (t_gen1 - t_gen0) * 1000.0
                print(f"[Accelerate] fwd: {fwd_ms:.2f} ms | gen({sample_tokens}): {gen_ms:.2f} ms")

                if tokenizer is not None:
                    text = _decode_tokens(tokenizer, gen_ids[0])
                    print("\n[Sample]\n", text)
                else:
                    print("\n[Sample token ids]\n", gen_ids[0].tolist())
        accelerator.wait_for_everyone()

    # Save checkpoint on main process after training
    if accelerator.is_local_main_process:
        torch.save({
            'model_state_dict': accelerator.unwrap_model(model).state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
        }, 'checkpoint.pt')
        print("[Accelerate] Checkpoint saved to checkpoint.pt")

        # After training, test KroneckerLinear materialize_dense if using Kronecker
        if config.use_kron:
            base_model = accelerator.unwrap_model(model)
            print("[Debug] Testing KroneckerLinear materialize_dense consistency...")
            for name, module in base_model.named_modules():
                if isinstance(module, KroneckerLinear):
                    max_diff = module.test_materialize_dense()
                    print(f"[Debug] {name}: max diff = {max_diff}")


def compare_kron_variants(base_config: GPTConfig, device: torch.device, batch_size: int, steps: int, gen_tokens: int, tokens: Optional[torch.Tensor] = None):
    variants = [("baseline", False), ("kron", True)]
    for name, use_kron in variants:
        cfg = replace(base_config, use_kron=use_kron)
        res = benchmark_variant(cfg, device, batch_size=batch_size, steps=steps, gen_tokens=gen_tokens, tokens=tokens)
        print(f"=== {name} ===")
        print(
            f"train avg: {res['train_ms']:.2f} ms/step | tokens/s: {res['train_tps']:.1f} | loss: {res['loss']:.4f}"
        )
        print(f"fwd: {res['fwd_ms']:.2f} ms | gen({gen_tokens}): {res['gen_ms']:.2f} ms")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--max_iters", type=int, default=50)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--compare_kron", action="store_true", help="Benchmark baseline vs Kronecker variants")
    parser.add_argument("--bench_steps", type=int, default=20, help="Number of steps for benchmark mode")
    parser.add_argument("--bench_gen_tokens", type=int, default=64, help="Generation length to time in benchmark mode")
    parser.add_argument("--use_kron", action="store_true", help="Use Kronecker-factored linears for attention/MLP")
    parser.add_argument("--kron_rank", type=int, default=1, help="Number of Kronecker factors to sum")
    parser.add_argument("--kron_infer_dense", action="store_true", help="Materialize dense weight at inference for single GEMM")
    parser.add_argument("--kron_fused", action="store_true", help="Use batched matmul fused path for Kronecker forward")
    parser.add_argument("--kron_triton_infer", action="store_true", help="Use Triton matmul on dense weight for inference-only path")
    parser.add_argument("--kron_triton_autograd", action="store_true", help="Use Triton matmul forward with torch backward (dense weight)")
    parser.add_argument("--kron_cuda_fused", action="store_true", help="Use CUDA extension fused forward/backward if built")
    parser.add_argument("--dataset", type=str, default="random", choices=["random", "shakespeare"], help="Training data source")
    parser.add_argument("--accelerate", action="store_true", help="Use HuggingFace Accelerate for multi-GPU training")
    parser.add_argument("--accelerate_steps", type=int, default=100, help="Number of steps when using Accelerate")
    parser.add_argument("--sample_after", action="store_true", help="After training, generate a sample")
    parser.add_argument("--sample_tokens", type=int, default=100, help="Number of new tokens to generate in the sample")
    parser.add_argument("--sample_prompt", type=str, default="To be, or not to be", help="Prompt text for sampling")
    parser.add_argument("--text_path", type=str, default=None, help="Path to local Shakespeare text (e.g., inputs.txt)")
    args = parser.parse_args()
    # If any kron_* flag is set, or --use_kron is passed, use_kron should be True
    kron_flags = [args.kron_fused, args.kron_triton_infer, args.kron_triton_autograd, args.kron_cuda_fused]
    use_kron = args.use_kron or any(kron_flags)
    config = GPTConfig(
        block_size=args.block_size,
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        bias=args.bias,
        use_kron=use_kron,
        kron_rank=args.kron_rank,
        kron_infer_dense=args.kron_infer_dense,
        kron_fused=args.kron_fused,
        kron_triton_infer=args.kron_triton_infer,
        kron_triton_autograd=args.kron_triton_autograd,
        kron_cuda_fused=args.kron_cuda_fused,
    )

    if args.accelerate:
        accelerate_train(
            config,
            batch_size=args.batch_size,
            steps=args.accelerate_steps,
            dataset=args.dataset,
            sample_after=args.sample_after,
            sample_tokens=args.sample_tokens,
            sample_prompt=args.sample_prompt,
            text_path=args.text_path,
        )
    else:
        device = torch.device(args.device)
        tokens = None
        if args.dataset == "shakespeare":
            _, train_ids, _ = load_shakespeare(text_path=args.text_path)
            tokens = train_ids.to(device)
        train(config, device, max_iters=args.max_iters, eval_interval=args.eval_interval, batch_size=args.batch_size, tokens=tokens)


if __name__ == "__main__":
    main()
