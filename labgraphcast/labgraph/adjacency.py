import numpy as np
import torch
from typing import List, Tuple

def build_slot_adjacency(slot_order: List[str], cyclic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Size]:
    """
    Build an undirected adjacency between consecutive slots.
    Returns torch.sparse_coo_tensor indices, values, and shape suitable for sparse matmul.
    """
    n = len(slot_order)
    edges = []
    for i in range(n - 1):
        edges.append((i, i + 1))
        edges.append((i + 1, i))
    if cyclic and n > 2:
        edges.append((0, n - 1))
        edges.append((n - 1, 0))
    # self-loops
    for i in range(n):
        edges.append((i, i))
    rows, cols = zip(*edges)
    idx = torch.tensor([rows, cols], dtype=torch.long)
    vals = torch.ones(len(edges), dtype=torch.float32)
    shape = torch.Size([n, n])
    return idx, vals, shape

def normalized_adj(idx: torch.Tensor, vals: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    Symmetric normalization: D^{-1/2} (A) D^{-1/2}
    """
    A = torch.sparse_coo_tensor(idx, vals, shape)
    deg = torch.sparse.sum(A, dim=1).to_dense()  # (n,)
    deg_inv_sqrt = torch.pow(deg.clamp(min=1.0), -0.5)
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    # Dense sandwich is fine for small S; for large S, switch to sparse ops.
    A_dense = A.to_dense()
    A_norm = D_inv_sqrt @ A_dense @ D_inv_sqrt
    return A_norm  # dense (n x n) for simplicity
