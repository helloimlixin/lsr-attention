"""
Test script for inference speed and accuracy comparison.

Compares:
- Dot-Product (SDPA/FlashAttention optimized)
- Dot-Product (naive, no FlashAttention)
- LSR (PyTorch)
- LSR (Triton optimized)
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import GPTLM
from attention import MultiHeadSelfAttention


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
    
    B, T, D, H, R = 2, 256, 512, 8, 16
    d_head = D // H
    
    x = torch.randn(B, T, D, device=device)
    
    # Create attention modules with same weights
    naive_attn = NaiveDotProductAttention(D, H).to(device).eval()
    sdpa_attn = MultiHeadSelfAttention(D, H, attn_type="dot").to(device).eval()
    lsr_attn = MultiHeadSelfAttention(D, H, attn_type="lsr", lsr_rank=R).to(device).eval()
    lsr_triton_attn = MultiHeadSelfAttention(D, H, attn_type="lsr_triton", lsr_rank=R).to(device).eval()
    
    # Copy weights from naive to sdpa
    sdpa_attn.load_state_dict(naive_attn.state_dict())
    
    # Copy LSR weights
    lsr_triton_attn.q_proj.load_state_dict(lsr_attn.q_proj.state_dict())
    lsr_triton_attn.k_proj.load_state_dict(lsr_attn.k_proj.state_dict())
    lsr_triton_attn.v_proj.load_state_dict(lsr_attn.v_proj.state_dict())
    lsr_triton_attn.o_proj.load_state_dict(lsr_attn.o_proj.state_dict())
    lsr_triton_attn.W_q_lsr.data = lsr_attn.W_q_lsr.data.clone()
    lsr_triton_attn.W_k_lsr.data = lsr_attn.W_k_lsr.data.clone()
    
    with torch.no_grad():
        out_naive = naive_attn(x)
        out_sdpa = sdpa_attn(x)
        out_lsr = lsr_attn(x)
        out_lsr_triton = lsr_triton_attn(x)
    
    print(f"\nComparing attention outputs (B={B}, T={T}, D={D}, H={H}):\n")
    
    # SDPA vs Naive (should be nearly identical)
    diff_sdpa = (out_sdpa - out_naive).abs().max().item()
    print(f"  SDPA vs Naive:           max diff = {diff_sdpa:.2e}  {'✓ PASS' if diff_sdpa < 1e-4 else '✗ FAIL'}")
    
    # LSR PyTorch vs LSR Triton (should be nearly identical)
    diff_lsr = (out_lsr_triton - out_lsr).abs().max().item()
    print(f"  LSR Triton vs LSR:       max diff = {diff_lsr:.2e}  {'✓ PASS' if diff_lsr < 1e-2 else '✗ FAIL'}")
    
    # LSR vs Naive (expected to be different - different attention pattern)
    diff_lsr_naive = (out_lsr - out_naive).abs().max().item()
    print(f"  LSR vs Naive (expected): max diff = {diff_lsr_naive:.2e}  (different attention)")
    
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
    
    # 3. LSR PyTorch
    for lsr_rank in [16, 32]:
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
        
        results.append((f"LSR PyTorch (R={lsr_rank})", lsr_time, naive_time / lsr_time))
    
    # 4. LSR Triton
    for lsr_rank in [16, 32]:
        model_triton = GPTLM(
            vocab_size, d_model=d_model, num_heads=num_heads,
            num_layers=num_layers, max_seq_len=seq_len,
            attn_type="lsr_triton", lsr_rank=lsr_rank
        ).to(device).eval()
        
        with torch.no_grad():
            for _ in range(warmup):
                _ = model_triton(x)
            torch.cuda.synchronize()
            
            t0 = time.time()
            for _ in range(iters):
                _ = model_triton(x)
            torch.cuda.synchronize()
            triton_time = (time.time() - t0) / iters * 1000
        
        results.append((f"LSR Triton (R={lsr_rank})", triton_time, naive_time / triton_time))
    
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
        ("LSR PyTorch (R=16)", GPTLM, {"attn_type": "lsr", "lsr_rank": 16}),
        ("LSR Triton (R=16)", GPTLM, {"attn_type": "lsr_triton", "lsr_rank": 16}),
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


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("LSR ATTENTION - COMPREHENSIVE TEST SUITE")
    print("=" * 70 + "\n")
    
    test_accuracy()
    test_speed()
    test_memory()
    
    print("All tests completed!")

