import torch

def einsum_dense(A, B, heads, head_dim, in_mult, out_mult):
    # Build the dense weight using the same contraction as the einsum path
    # A: (out_mult*heads, in_mult*heads)
    # B: (head_dim, head_dim)
    # Output: (out_mult*heads*head_dim, in_mult*heads*head_dim)
    OM, IM = A.shape
    hd = B.shape[0]
    W = torch.zeros(OM*hd, IM*hd)
    for i in range(OM):
        for j in range(IM):
            # Place B * A[i, j] in the correct block
            W[i*hd:(i+1)*hd, j*hd:(j+1)*hd] = B * A[i, j]
    return W

def test_block_vs_einsum():
    heads = 2
    head_dim = 3
    in_mult = 1
    out_mult = 1
    rank = 1
    A = torch.randn(rank, out_mult * heads, in_mult * heads)
    B = torch.randn(rank, head_dim, head_dim)
    x = torch.randn(1, heads * head_dim * in_mult)

    # Correct dense path (block-wise)
    W = sum(einsum_dense(A[r], B[r], heads, head_dim, in_mult, out_mult) for r in range(rank))
    out_dense = x @ W.t()

    # Einsum path (matches your forward)
    x_flat = x.view(-1, head_dim, in_mult * heads)
    mid = torch.einsum("bdi,roi->bdro", x_flat, A)
    y = torch.einsum("rde,bdro->bero", B, mid)
    acc = y.sum(dim=2)
    out_einsum = acc.transpose(1, 2).contiguous().view(-1, out_mult * heads * head_dim)

    print("Dense output:", out_dense)
    print("Einsum output:", out_einsum)
    print("Max diff:", (out_dense - out_einsum).abs().max().item())

if __name__ == "__main__":
    test_block_vs_einsum()
