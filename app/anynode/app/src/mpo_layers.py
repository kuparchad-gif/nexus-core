from typing import List, Optional
import math
import torch
from torch import nn
import torch.nn.functional as F

def _balanced_factors(dim: int, n_cores: int) -> List[int]:
    root = round(dim ** (1 / n_cores))
    factors = [root] * n_cores
    prod = math.prod(factors)
    i = 0
    while prod < dim:
        factors[i % n_cores] += 1
        prod = math.prod(factors); i += 1
    i = n_cores - 1
    while prod > dim:
        if factors[i] > 1:
            factors[i] -= 1
            prod = math.prod(factors)
        i = (i - 1) % n_cores
    if prod < dim:
        factors[-1] += (dim - prod)
    assert math.prod(factors) == dim
    return factors

def _svd_truncate(M: torch.Tensor, rank: int):
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    r = min(rank, S.numel())
    return U[:, :r], S[:r], Vh[:r, :]

class MPOLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 n_cores: int = 4, chi_list: Optional[List[int]] = None,
                 bias: bool = True, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_cores = n_cores

        self.in_modes  = _balanced_factors(in_features, n_cores)
        self.out_modes = _balanced_factors(out_features, n_cores)
        self.ranks = [1] + (chi_list if chi_list is not None else [64] * (n_cores - 1)) + [1]
        assert len(self.ranks) == n_cores + 1

        cores = []
        for k in range(n_cores):
            rk_1, rk = self.ranks[k], self.ranks[k+1]
            ok, dk = self.out_modes[k], self.in_modes[k]
            G = nn.Parameter(torch.zeros(rk_1, ok, dk, rk, **factory_kwargs))
            nn.init.kaiming_uniform_(G.view(rk_1, ok*dk*rk), a=math.sqrt(5))
            cores.append(G)
        self.cores = nn.ParameterList(cores)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self._dense_cache: Optional[torch.Tensor] = None
        self.register_buffer("_cache_version", torch.tensor(0, dtype=torch.long))
        self.register_buffer("_last_materialized", torch.tensor(-1, dtype=torch.long))

    @torch.no_grad()
    def init_from_dense(self, W: torch.Tensor, max_rank: int):
        assert W.shape == (self.out_features, self.in_features)
        T = W.reshape(*self.out_modes, *self.in_modes)
        left_rank = 1
        res = T
        ranks = [1]
        cores = []
        for k in range(self.n_cores - 1):
            o_k = self.out_modes[k]; d_k = self.in_modes[k]
            left_dim = left_rank * o_k * d_k
            right_dim = int(res.numel() // left_dim)
            res = res.reshape(left_dim, right_dim)
            U, S, Vh = _svd_truncate(res, max_rank)
            r_k = U.shape[1]
            ranks.append(r_k)
            Gk = U.reshape(left_rank, o_k, d_k, r_k)
            cores.append(Gk)
            res = (S.unsqueeze(1) * Vh).reshape(r_k, -1)
            left_rank = r_k
        o_k = self.out_modes[-1]; d_k = self.in_modes[-1]
        G_last = res.reshape(left_rank, o_k, d_k, 1)
        ranks.append(1)
        self.ranks = ranks
        with torch.no_grad():
            for p, g in zip(self.cores[:-1], cores):
                p.copy_(g)
            self.cores[-1].copy_(G_last)
        self._cache_version += 1

    def materialize_weight(self) -> torch.Tensor:
        if (self._dense_cache is not None) and (self._last_materialized.item() == self._cache_version.item()):
            return self._dense_cache
        W = self.cores[0]
        for k in range(1, self.n_cores):
            W = torch.tensordot(W, self.cores[k], dims=([3],[0]))
        W = W.squeeze(0).squeeze(-1)  # (o0,d0,o1,d1,...)
        n = self.n_cores
        perm = list(range(0, 2*n, 2)) + list(range(1, 2*n, 2))  # o... then d...
        W = W.permute(perm).contiguous()
        out_dim = 1
        for k in range(n): out_dim *= self.out_modes[k]
        in_dim  = 1
        for k in range(n): in_dim  *= self.in_modes[k]
        W = W.reshape(out_dim, in_dim)
        self._dense_cache = W
        self._last_materialized = self._cache_version.clone()
        return W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.materialize_weight()
        return F.linear(x, W, self.bias)
