"""
Test script for inference speed and accuracy comparison.

Compares:
- Dot-Product (SDPA/FlashAttention optimized)
- Dot-Product (naive, no FlashAttention)
- LSR (Triton optimized with online softmax)
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import GPTLM
from attention import MultiHeadSelfAttention, scaled_dot_product_attention
from lsr_triton_parallel import lsr_attention_online, lsr_attention_fused


class NaiveDotProductAttention(nn.Module):
    """
    Naive dot-product attention WITHOUT FlashAttention optimization.
    Materializes the full T x T attention matrix.
    """

    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)

        # Naive attention: explicitly compute T x T scores
        scale = 1.0 / math.sqrt(self.d_head)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, T, T)

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        y = torch.matmul(attn, v)

        y = y.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.o_proj(y)


class DecoderBlockNaive(nn.Module):
    """Decoder block with naive dot-product attention."""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = NaiveDotProductAttention(d_model, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPTLMNaive(nn.Module):
    """GPT with naive (non-FlashAttention) dot-product attention."""

    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=8,
                 d_ff=2048, max_seq_len=256, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList([
            DecoderBlockNaive(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx, targets=None):
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
        return logits, None


def test_accuracy():
    """Test numerical accuracy of different attention implementations."""
    print("=" * 70)
    print("ACCURACY TEST")
    print("=" * 70)

    torch.manual_seed(42)
    device = "cuda"

    B, T, D, H, R = 2, 256, 512, 8, 8
    d_head = D // H

    x = torch.randn(B, T, D, device=device)

    # Create attention modules with same weights
    naive_attn = NaiveDotProductAttention(D, H).to(device).eval()
    sdpa_attn = MultiHeadSelfAttention(D, H, attn_type="dot").to(device).eval()
    lsr_attn = MultiHeadSelfAttention(D, H, attn_type="lsr", lsr_rank=R).to(device).eval()

    # Copy weights from naive to sdpa
    sdpa_attn.load_state_dict(naive_attn.state_dict())

    with torch.no_grad():
        out_naive = naive_attn(x)
        out_sdpa = sdpa_attn(x)
        out_lsr = lsr_attn(x)

    print(f"\nComparing attention outputs (B={B}, T={T}, D={D}, H={H}):\n")

    # SDPA vs Naive (should be nearly identical)
    diff_sdpa = (out_sdpa - out_naive).abs().max().item()
    print(f"  SDPA vs Naive:           max diff = {diff_sdpa:.2e}  {'PASS' if diff_sdpa < 1e-4 else 'FAIL'}")

    # LSR vs Naive (expected to be different - different attention pattern)
    diff_lsr_naive = (out_lsr - out_naive).abs().max().item()
    print(f"  LSR vs Naive (expected): max diff = {diff_lsr_naive:.2e}  (different attention)")

    # Test standalone kernel accuracy
    print(f"\nStandalone kernel accuracy test:")
    q = torch.randn(B, H, T, d_head, device=device, dtype=torch.float32)
    k = torch.randn(B, H, T, d_head, device=device, dtype=torch.float32)
    v = torch.randn(B, H, T, d_head, device=device, dtype=torch.float32)
    W_q = torch.randn(H, d_head, R, device=device, dtype=torch.float32) / math.sqrt(d_head)
    W_k = torch.randn(H, d_head, R, device=device, dtype=torch.float32) / math.sqrt(d_head)
    core = torch.ones(H, R, device=device, dtype=torch.float32)

    # Reference: naive LSR computation
    def lsr_reference(q, k, v, W_q, W_k, core):
        q_lr = torch.einsum("bhtd,hdr->bhtr", q, W_q)
        k_lr = torch.einsum("bhtd,hdr->bhtr", k, W_k)
        scale = 1.0 / math.sqrt(R)
        scores = torch.einsum("bhir,hr,bhjr->bhij", q_lr, core, k_lr) * scale
        causal_mask = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        return torch.einsum("bhij,bhjd->bhid", attn, v)

    with torch.no_grad():
        out_ref = lsr_reference(q, k, v, W_q, W_k, core)
        out_triton = lsr_attention_online(q, k, v, W_q, W_k, core, causal=True)

    diff_triton_vs_ref = (out_triton - out_ref).abs().max().item()
    print(f"  LSR Triton vs Reference: max diff = {diff_triton_vs_ref:.2e}  {'PASS' if diff_triton_vs_ref < 1e-2 else 'FAIL'}")

    print()


def test_speed():
    """Benchmark inference speed of different attention implementations."""
    print("=" * 70)
    print("SPEED TEST")
    print("=" * 70)

    device = "cuda"
    vocab_size = 50257
    batch_size = 4
    seq_len = 1024
    num_layers = 4
    d_model = 512
    num_heads = 8
    warmup = 5
    iters = 20

    print(f"\nConfig: batch={batch_size}, seq_len={seq_len}, layers={num_layers}, d_model={d_model}\n")

    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    results = []

    # 1. Naive Dot-Product (no FlashAttention)
    model_naive = GPTLMNaive(
        vocab_size, d_model=d_model, num_heads=num_heads,
        num_layers=num_layers, max_seq_len=seq_len
    ).to(device).eval()

    with torch.no_grad():
        for _ in range(warmup):
            _ = model_naive(x)
        torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(iters):
            _ = model_naive(x)
        torch.cuda.synchronize()
        naive_time = (time.time() - t0) / iters * 1000

    results.append(("Dot-Product (Naive)", naive_time, 1.0))

    # 2. SDPA Dot-Product (FlashAttention)
    model_sdpa = GPTLM(
        vocab_size, d_model=d_model, num_heads=num_heads,
        num_layers=num_layers, max_seq_len=seq_len, attn_type="dot"
    ).to(device).eval()

    with torch.no_grad():
        for _ in range(warmup):
            _ = model_sdpa(x)
        torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(iters):
            _ = model_sdpa(x)
        torch.cuda.synchronize()
        sdpa_time = (time.time() - t0) / iters * 1000

    results.append(("Dot-Product (SDPA/Flash)", sdpa_time, naive_time / sdpa_time))

    # 3. LSR Triton (online softmax)
    for lsr_rank in [4, 8]:
        model_lsr = GPTLM(
            vocab_size, d_model=d_model, num_heads=num_heads,
            num_layers=num_layers, max_seq_len=seq_len,
            attn_type="lsr", lsr_rank=lsr_rank
        ).to(device).eval()

        with torch.no_grad():
            for _ in range(warmup):
                _ = model_lsr(x)
            torch.cuda.synchronize()

            t0 = time.time()
            for _ in range(iters):
                _ = model_lsr(x)
            torch.cuda.synchronize()
            lsr_time = (time.time() - t0) / iters * 1000

        results.append((f"LSR Triton (R={lsr_rank})", lsr_time, naive_time / lsr_time))

    # 4. LSR Fused (QKV+attention+output in single kernel)
    for lsr_rank in [4, 8]:
        model_fused = GPTLM(
            vocab_size, d_model=d_model, num_heads=num_heads,
            num_layers=num_layers, max_seq_len=seq_len,
            attn_type="lsr_fused", lsr_rank=lsr_rank
        ).to(device).eval()

        with torch.no_grad():
            for _ in range(warmup):
                _ = model_fused(x)
            torch.cuda.synchronize()

            t0 = time.time()
            for _ in range(iters):
                _ = model_fused(x)
            torch.cuda.synchronize()
            fused_time = (time.time() - t0) / iters * 1000

        results.append((f"LSR Fused (R={lsr_rank})", fused_time, naive_time / fused_time))

    # Print results
    print(f"{'Attention Type':<28} {'Time (ms)':<12} {'Speedup':<10}")
    print("-" * 50)
    for name, time_ms, speedup in results:
        print(f"{name:<28} {time_ms:>8.2f} ms   {speedup:>5.2f}x")

    print()

    # Find fastest
    fastest = min(results, key=lambda x: x[1])
    print(f"Fastest: {fastest[0]} ({fastest[1]:.2f} ms)")
    print()


def test_memory():
    """Test peak memory usage of different attention implementations."""
    print("=" * 70)
    print("MEMORY TEST")
    print("=" * 70)

    device = "cuda"
    vocab_size = 50257
    batch_size = 4
    seq_len = 1024
    num_layers = 4
    d_model = 512
    num_heads = 8

    print(f"\nConfig: batch={batch_size}, seq_len={seq_len}, layers={num_layers}\n")

    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    results = []

    for name, model_cls, kwargs in [
        ("Dot-Product (Naive)", GPTLMNaive, {}),
        ("Dot-Product (SDPA)", GPTLM, {"attn_type": "dot"}),
        ("LSR Triton (R=4)", GPTLM, {"attn_type": "lsr", "lsr_rank": 4}),
        ("LSR Triton (R=8)", GPTLM, {"attn_type": "lsr", "lsr_rank": 8}),
        ("LSR Fused (R=4)", GPTLM, {"attn_type": "lsr_fused", "lsr_rank": 4}),
        ("LSR Fused (R=8)", GPTLM, {"attn_type": "lsr_fused", "lsr_rank": 8}),
    ]:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        model = model_cls(
            vocab_size, d_model=d_model, num_heads=num_heads,
            num_layers=num_layers, max_seq_len=seq_len, **kwargs
        ).to(device).eval()

        with torch.no_grad():
            _ = model(x)

        peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        results.append((name, peak_mb))

        del model
        torch.cuda.empty_cache()

    print(f"{'Attention Type':<28} {'Peak Memory':<15}")
    print("-" * 45)
    for name, mem in results:
        print(f"{name:<28} {mem:>10.1f} MB")

    print()


def test_attention_kernels():
    """Benchmark attention kernels directly (without full model overhead)."""
    print("=" * 70)
    print("ATTENTION KERNEL SPEED TEST (Standalone)")
    print("=" * 70)

    device = "cuda"
    warmup = 10
    iters = 50

    configs = [
        # (B, H, T, D, R)
        (4, 8, 512, 64, 4),
        (4, 8, 1024, 64, 4),
        (4, 8, 2048, 64, 4),
        (4, 8, 1024, 64, 8),
    ]

    for B, H, T, D, R in configs:
        print(f"\nConfig: B={B}, H={H}, T={T}, D={D}, R={R}")
        print("-" * 50)

        # Create inputs
        q = torch.randn(B, H, T, D, device=device, dtype=torch.float32)
        k = torch.randn(B, H, T, D, device=device, dtype=torch.float32)
        v = torch.randn(B, H, T, D, device=device, dtype=torch.float32)
        W_q = torch.randn(H, D, R, device=device, dtype=torch.float32) / math.sqrt(D)
        W_k = torch.randn(H, D, R, device=device, dtype=torch.float32) / math.sqrt(D)
        core = torch.ones(H, R, device=device, dtype=torch.float32)

        results = []

        # 1. Naive Dot-Product (materializes T x T)
        def naive_dot():
            scale = 1.0 / math.sqrt(D)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            causal_mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(causal_mask, float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            return torch.matmul(attn, v)

        with torch.no_grad():
            for _ in range(warmup):
                _ = naive_dot()
            torch.cuda.synchronize()

            t0 = time.time()
            for _ in range(iters):
                _ = naive_dot()
            torch.cuda.synchronize()
            naive_time = (time.time() - t0) / iters * 1000

        results.append(("Dot-Product (Naive)", naive_time))

        # 2. SDPA (PyTorch optimized)
        with torch.no_grad():
            for _ in range(warmup):
                _ = scaled_dot_product_attention(q, k, v, causal=True)
            torch.cuda.synchronize()

            t0 = time.time()
            for _ in range(iters):
                _ = scaled_dot_product_attention(q, k, v, causal=True)
            torch.cuda.synchronize()
            sdpa_time = (time.time() - t0) / iters * 1000

        results.append(("Dot-Product (SDPA)", sdpa_time))

        # 3. LSR Triton (online softmax)
        with torch.no_grad():
            for _ in range(warmup):
                _ = lsr_attention_online(q, k, v, W_q, W_k, core, causal=True)
            torch.cuda.synchronize()

            t0 = time.time()
            for _ in range(iters):
                _ = lsr_attention_online(q, k, v, W_q, W_k, core, causal=True)
            torch.cuda.synchronize()
            lsr_time = (time.time() - t0) / iters * 1000

        results.append(("LSR Triton (online)", lsr_time))

        # 4. LSR Fused (QKV + attention + output in single kernel)
        d_model = H * D
        x_input = torch.randn(B, T, d_model, device=device, dtype=torch.float32)
        W_qkv = torch.randn(d_model, 3 * d_model, device=device, dtype=torch.float32) / math.sqrt(d_model)
        W_o = torch.randn(d_model, d_model, device=device, dtype=torch.float32) / math.sqrt(d_model)

        with torch.no_grad():
            for _ in range(warmup):
                _ = lsr_attention_fused(x_input, W_qkv, W_q, W_k, core, W_o, causal=True)
            torch.cuda.synchronize()

            t0 = time.time()
            for _ in range(iters):
                _ = lsr_attention_fused(x_input, W_qkv, W_q, W_k, core, W_o, causal=True)
            torch.cuda.synchronize()
            fused_time = (time.time() - t0) / iters * 1000

        results.append(("LSR Fused (QKV+attn+O)", fused_time))

        # Print results
        baseline = results[0][1]  # Naive dot-product as baseline
        print(f"{'Attention Type':<24} {'Time (ms)':<12} {'vs Naive':<10} {'vs SDPA':<10}")
        print("-" * 56)
        for name, time_ms in results:
            speedup_naive = baseline / time_ms
            speedup_sdpa = sdpa_time / time_ms
            print(f"{name:<24} {time_ms:>8.2f} ms   {speedup_naive:>5.2f}x     {speedup_sdpa:>5.2f}x")

        # Find fastest
        fastest = min(results, key=lambda x: x[1])
        print(f"\nFastest: {fastest[0]} ({fastest[1]:.2f} ms)")

    print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("LSR ATTENTION - COMPREHENSIVE TEST SUITE")
    print("=" * 70 + "\n")

    test_accuracy()
    test_attention_kernels()
    test_speed()
    test_memory()

    print("All tests completed!")
