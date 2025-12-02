"""Benchmark script comparing original vs parallelized LSR attention kernels."""

import math
import time
import torch

# Original implementations
from lsr_triton import (
    lsr_attention_triton,
    lsr_attention_triton_factorized,
)

# New parallelized implementations
from lsr_triton_parallel import (
    lsr_attention_online,
    lsr_attention_kronecker_parallel,
    _run_kronecker_cached,
)

# PyTorch reference
from attention import lsr_attention


def batched_kron(a, b):
    """Compute Kronecker product per row: (H, R1), (H, R2) -> (H, R1*R2)."""
    # a: (H, R1), b: (H, R2) -> (H, R1*R2)
    H, R1 = a.shape
    _, R2 = b.shape
    # Outer product per head: a[:, :, None] * b[:, None, :] -> (H, R1, R2)
    # Then flatten last two dims
    return (a[:, :, None] * b[:, None, :]).reshape(H, R1 * R2)


def benchmark_kernel(fn, args, warmup=10, iters=50, name="kernel"):
    """Benchmark a kernel function."""
    # Warmup
    for _ in range(warmup):
        _ = fn(*args)
    torch.cuda.synchronize()

    # Benchmark
    t0 = time.time()
    for _ in range(iters):
        _ = fn(*args)
    torch.cuda.synchronize()
    elapsed = (time.time() - t0) / iters * 1000  # ms

    return elapsed


def test_correctness():
    """Test that new kernels produce correct results."""
    print("=" * 70)
    print("CORRECTNESS TEST")
    print("=" * 70)

    torch.manual_seed(42)
    device = "cuda"

    B, H, T, D, R = 2, 8, 256, 64, 8

    q = torch.randn(B, H, T, D, device=device, dtype=torch.float32)
    k = torch.randn(B, H, T, D, device=device, dtype=torch.float32)
    v = torch.randn(B, H, T, D, device=device, dtype=torch.float32)
    W_q = torch.randn(H, D, R, device=device, dtype=torch.float32) / math.sqrt(D)
    W_k = torch.randn(H, D, R, device=device, dtype=torch.float32) / math.sqrt(D)
    core = torch.ones(H, R, device=device, dtype=torch.float32)

    print(f"\nFlat core test (B={B}, H={H}, T={T}, D={D}, R={R}):")

    with torch.no_grad():
        out_pytorch = lsr_attention(q, k, v, W_q, W_k, core, causal=True)
        out_triton = lsr_attention_triton(q, k, v, W_q, W_k, core, causal=True)
        out_online = lsr_attention_online(q, k, v, W_q, W_k, core, causal=True)

    diff_triton = (out_triton - out_pytorch).abs().max().item()
    diff_online = (out_online - out_pytorch).abs().max().item()

    print(f"  Original Triton vs PyTorch: max diff = {diff_triton:.2e} {'PASS' if diff_triton < 1e-2 else 'FAIL'}")
    print(f"  Online Triton vs PyTorch:   max diff = {diff_online:.2e} {'PASS' if diff_online < 1e-2 else 'FAIL'}")

    # Kronecker factorized test - R2 must be >= 16 for Triton dot product
    R1, R2 = 2, 16
    R_kron = R1 * R2
    W_q_kron = torch.randn(H, D, R_kron, device=device, dtype=torch.float32) / math.sqrt(D)
    W_k_kron = torch.randn(H, D, R_kron, device=device, dtype=torch.float32) / math.sqrt(D)
    core1 = torch.ones(H, R1, device=device, dtype=torch.float32)
    core2 = torch.ones(H, R2, device=device, dtype=torch.float32)

    print(f"\nKronecker test (R1={R1}, R2={R2}, total R={R_kron}):")

    with torch.no_grad():
        # Reference: expand Kronecker and use flat implementation
        core_flat = batched_kron(core1, core2)
        out_ref = lsr_attention(q, k, v, W_q_kron, W_k_kron, core_flat, causal=True)

        out_kron_parallel = lsr_attention_kronecker_parallel(
            q, k, v, W_q_kron, W_k_kron, core1, core2, causal=True
        )

    diff_kron_parallel = (out_kron_parallel - out_ref).abs().max().item()
    print(f"  Kronecker Parallel vs Ref:  max diff = {diff_kron_parallel:.2e} {'PASS' if diff_kron_parallel < 1e-2 else 'FAIL'}")

    print()


def benchmark_flat_core():
    """Benchmark flat core implementations."""
    print("=" * 70)
    print("FLAT CORE BENCHMARK (Online Softmax)")
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
        (4, 8, 1024, 64, 16),
    ]

    for B, H, T, D, R in configs:
        print(f"\nConfig: B={B}, H={H}, T={T}, D={D}, R={R}")
        print("-" * 60)

        q = torch.randn(B, H, T, D, device=device, dtype=torch.float32)
        k = torch.randn(B, H, T, D, device=device, dtype=torch.float32)
        v = torch.randn(B, H, T, D, device=device, dtype=torch.float32)
        W_q = torch.randn(H, D, R, device=device, dtype=torch.float32) / math.sqrt(D)
        W_k = torch.randn(H, D, R, device=device, dtype=torch.float32) / math.sqrt(D)
        core = torch.ones(H, R, device=device, dtype=torch.float32)

        results = []

        # PyTorch reference
        t_pytorch = benchmark_kernel(
            lambda: lsr_attention(q, k, v, W_q, W_k, core, causal=True),
            [], warmup=warmup, iters=iters, name="PyTorch"
        )
        results.append(("LSR PyTorch", t_pytorch))

        # Original Triton (two-pass)
        t_triton = benchmark_kernel(
            lambda: lsr_attention_triton(q, k, v, W_q, W_k, core, causal=True),
            [], warmup=warmup, iters=iters, name="Triton Original"
        )
        results.append(("LSR Triton (two-pass)", t_triton))

        # Online Triton (single-pass)
        t_online = benchmark_kernel(
            lambda: lsr_attention_online(q, k, v, W_q, W_k, core, causal=True),
            [], warmup=warmup, iters=iters, name="Triton Online"
        )
        results.append(("LSR Triton (online)", t_online))

        # Print results
        baseline = results[0][1]
        print(f"{'Implementation':<28} {'Time (ms)':<12} {'vs PyTorch':<12} {'vs Original':<12}")
        print("-" * 64)
        for name, t in results:
            speedup_pytorch = baseline / t
            speedup_original = t_triton / t
            print(f"{name:<28} {t:>8.3f} ms   {speedup_pytorch:>8.2f}x    {speedup_original:>8.2f}x")

        print(f"\nOnline softmax speedup over two-pass: {t_triton / t_online:.2f}x")


def benchmark_kronecker():
    """Benchmark Kronecker factorized implementations."""
    print("\n" + "=" * 70)
    print("KRONECKER FACTORIZED BENCHMARK")
    print("=" * 70)

    device = "cuda"
    warmup = 10
    iters = 50

    configs = [
        # (B, H, T, D, R1, R2) - R2 must be >= 16 for Triton dot product
        (4, 8, 512, 64, 2, 16),
        (4, 8, 1024, 64, 2, 16),
        (4, 8, 1024, 64, 4, 16),
        (4, 8, 2048, 64, 2, 16),
    ]

    for B, H, T, D, R1, R2 in configs:
        R = R1 * R2
        print(f"\nConfig: B={B}, H={H}, T={T}, D={D}, R1={R1}, R2={R2} (total R={R})")
        print("-" * 70)

        q = torch.randn(B, H, T, D, device=device, dtype=torch.float32)
        k = torch.randn(B, H, T, D, device=device, dtype=torch.float32)
        v = torch.randn(B, H, T, D, device=device, dtype=torch.float32)
        W_q = torch.randn(H, D, R, device=device, dtype=torch.float32) / math.sqrt(D)
        W_k = torch.randn(H, D, R, device=device, dtype=torch.float32) / math.sqrt(D)
        core1 = torch.ones(H, R1, device=device, dtype=torch.float32)
        core2 = torch.ones(H, R2, device=device, dtype=torch.float32)
        core_flat = batched_kron(core1, core2)

        results = []

        # PyTorch reference (flat)
        t_pytorch = benchmark_kernel(
            lambda: lsr_attention(q, k, v, W_q, W_k, core_flat, causal=True),
            [], warmup=warmup, iters=iters
        )
        results.append(("LSR PyTorch (flat)", t_pytorch))

        # Flat core with online softmax
        t_online = benchmark_kernel(
            lambda: lsr_attention_online(q, k, v, W_q, W_k, core_flat, causal=True),
            [], warmup=warmup, iters=iters
        )
        results.append(("LSR Triton Online (flat)", t_online))

        # Kronecker parallel (new fused kernel)
        t_kron_parallel = benchmark_kernel(
            lambda: lsr_attention_kronecker_parallel(q, k, v, W_q, W_k, core1, core2, causal=True),
            [], warmup=warmup, iters=iters
        )
        results.append(("LSR Kronecker Parallel", t_kron_parallel))

        # Print results
        print(f"{'Implementation':<32} {'Time (ms)':<12} {'vs PyTorch':<12}")
        print("-" * 56)
        for name, t in results:
            speedup_pytorch = t_pytorch / t
            print(f"{name:<32} {t:>8.3f} ms   {speedup_pytorch:>8.2f}x")

        print(f"\nKronecker parallel speedup over PyTorch: {t_pytorch / t_kron_parallel:.2f}x")


def benchmark_backward():
    """Benchmark backward pass (gradient computation)."""
    print("\n" + "=" * 70)
    print("BACKWARD PASS BENCHMARK")
    print("=" * 70)

    device = "cuda"
    warmup = 5
    iters = 20

    B, H, T, D, R = 4, 8, 1024, 64, 8

    print(f"\nConfig: B={B}, H={H}, T={T}, D={D}, R={R}")
    print("-" * 60)

    def run_backward(attn_fn, q, k, v, W_q, W_k, core):
        q = q.clone().requires_grad_(True)
        k = k.clone().requires_grad_(True)
        v = v.clone().requires_grad_(True)
        W_q = W_q.clone().requires_grad_(True)
        W_k = W_k.clone().requires_grad_(True)
        core = core.clone().requires_grad_(True)

        out = attn_fn(q, k, v, W_q, W_k, core, causal=True)
        loss = out.sum()
        loss.backward()
        return q.grad, k.grad, v.grad

    q = torch.randn(B, H, T, D, device=device, dtype=torch.float32)
    k = torch.randn(B, H, T, D, device=device, dtype=torch.float32)
    v = torch.randn(B, H, T, D, device=device, dtype=torch.float32)
    W_q = torch.randn(H, D, R, device=device, dtype=torch.float32) / math.sqrt(D)
    W_k = torch.randn(H, D, R, device=device, dtype=torch.float32) / math.sqrt(D)
    core = torch.ones(H, R, device=device, dtype=torch.float32)

    results = []

    # Original Triton
    for _ in range(warmup):
        run_backward(lsr_attention_triton, q, k, v, W_q, W_k, core)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        run_backward(lsr_attention_triton, q, k, v, W_q, W_k, core)
    torch.cuda.synchronize()
    t_triton = (time.time() - t0) / iters * 1000
    results.append(("Triton (two-pass)", t_triton))

    # Online Triton
    for _ in range(warmup):
        run_backward(lsr_attention_online, q, k, v, W_q, W_k, core)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        run_backward(lsr_attention_online, q, k, v, W_q, W_k, core)
    torch.cuda.synchronize()
    t_online = (time.time() - t0) / iters * 1000
    results.append(("Triton (online)", t_online))

    print(f"{'Implementation':<28} {'Fwd+Bwd (ms)':<15} {'Speedup':<10}")
    print("-" * 55)
    baseline = results[0][1]
    for name, t in results:
        speedup = baseline / t
        print(f"{name:<28} {t:>10.3f} ms    {speedup:>6.2f}x")


def memory_comparison():
    """Compare memory usage."""
    print("\n" + "=" * 70)
    print("MEMORY USAGE COMPARISON")
    print("=" * 70)

    device = "cuda"
    B, H, T, D, R = 4, 8, 2048, 64, 8

    print(f"\nConfig: B={B}, H={H}, T={T}, D={D}, R={R}")
    print("-" * 50)

    q = torch.randn(B, H, T, D, device=device, dtype=torch.float32)
    k = torch.randn(B, H, T, D, device=device, dtype=torch.float32)
    v = torch.randn(B, H, T, D, device=device, dtype=torch.float32)
    W_q = torch.randn(H, D, R, device=device, dtype=torch.float32) / math.sqrt(D)
    W_k = torch.randn(H, D, R, device=device, dtype=torch.float32) / math.sqrt(D)
    core = torch.ones(H, R, device=device, dtype=torch.float32)

    results = []

    for name, fn in [
        ("Triton (two-pass)", lsr_attention_triton),
        ("Triton (online)", lsr_attention_online),
    ]:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        with torch.no_grad():
            _ = fn(q, k, v, W_q, W_k, core, causal=True)

        peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        results.append((name, peak_mb))

    print(f"{'Implementation':<28} {'Peak Memory':<15}")
    print("-" * 45)
    for name, mem in results:
        print(f"{name:<28} {mem:>10.1f} MB")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("LSR ATTENTION PARALLELIZATION BENCHMARK")
    print("=" * 70 + "\n")

    test_correctness()
    benchmark_flat_core()
    benchmark_kronecker()
    benchmark_backward()
    memory_comparison()

    print("\nBenchmark complete!")
