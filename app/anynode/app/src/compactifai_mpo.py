# compactifai_mpo.py
from typing import Optional
import torch
import torch.nn as nn

def pick_rank(in_f: int, out_f: int, rank_max: int, rank_frac: float) -> int:
    base = int(max(1, round(min(in_f, out_f) * rank_frac)))
    return int(min(rank_max, base))

@torch.no_grad()
def svd_compress_linear(lin: nn.Linear, rank: int) -> nn.Sequential:
    W = lin.weight.data
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]

    L1 = nn.Linear(Vh_r.shape[1], rank, bias=False)
    L2 = nn.Linear(rank, U_r.shape[0], bias=True)

    L1.weight.data.copy_(Vh_r.T.contiguous())
    L2.weight.data.copy_((U_r * S_r.unsqueeze(0)).contiguous())
    if lin.bias is not None:
        L2.bias.data.copy_(lin.bias.data.to(W.device))
    else:
        nn.init.zeros_(L2.bias)

    return nn.Sequential(L1, L2)

class LayerCompressor:
    def __init__(self, rank_max:int=192, rank_frac:float=0.12):
        self.rank_max = rank_max
        self.rank_frac = rank_frac

    def compress_linear(self, lin: nn.Linear) -> nn.Module:
        in_f = lin.in_features
        out_f = lin.out_features
        rank = pick_rank(in_f, out_f, self.rank_max, self.rank_frac)
        if rank < 1 or rank >= min(in_f, out_f):
            return lin
        return svd_compress_linear(lin, rank)