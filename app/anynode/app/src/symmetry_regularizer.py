from typing import List
import torch

def build_ring_permutations(ring_size: int = 12, include_reflections: bool = True) -> List[torch.Tensor]:
    perms = []
    base = torch.arange(ring_size)
    for r in range(1, 6):
        perms.append((base + r) % ring_size)
    if include_reflections:
        perms.append(torch.flip(base, dims=[0]))
    return perms

def _apply_perm_to_matrix(W: torch.Tensor, prow: torch.Tensor, pcol: torch.Tensor) -> torch.Tensor:
    O, I = W.shape
    prow_full = prow.repeat(O // len(prow) + 1)[:O]
    pcol_full = pcol.repeat(I // len(pcol) + 1)[:I]
    return W.index_select(0, prow_full).index_select(1, pcol_full)

def symmetry_loss(W: torch.Tensor, perms: List[torch.Tensor], alpha: float = 0.01) -> torch.Tensor:
    if W.ndim != 2 or not perms:
        return W.new_zeros(())
    loss = W.new_zeros(())
    for p in perms:
        Wg = _apply_perm_to_matrix(W, p, p)
        loss = loss + torch.mean((W - Wg) ** 2)
    return alpha * loss
