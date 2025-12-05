import torch
import torch.autograd.profiler as profiler
from mingpt import KroneckerLinear

# Adjust these to match your model config
B = 64
heads = 4
head_dim = 64
in_mult = 1
out_mult = 1
rank = 4

layer = KroneckerLinear(heads, head_dim, in_mult, out_mult, rank, bias=False, cuda_fused=True).cuda()
x = torch.randn(B, heads * head_dim * in_mult, device='cuda')

# Warmup
for _ in range(10):
    y = layer(x)
    torch.cuda.synchronize()

# Profile
with profiler.profile(use_cuda=True, record_shapes=True) as prof:
    for _ in range(20):
        y = layer(x)
        torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
