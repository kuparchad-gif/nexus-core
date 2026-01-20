from typing import List
import numpy as np

def allocate_chi_per_core(
    n_cores: int,
    chi_base: int,
    degrees: List[float],
    fib_weights: List[float],
    centralities: List[float],
    chi_min: int = 32,
) -> List[int]:
    def _fit(v, n):
        v = np.array(v, dtype=np.float64)
        if len(v) < n:
            reps = n // len(v) + 1
            v = np.tile(v, reps)[:n]
        elif len(v) > n:
            v = v[:n]
        return v
    deg = _fit(degrees, n_cores)
    fib = _fit(fib_weights, n_cores)
    cen = _fit(centralities, n_cores)
    mix = 0.5 * deg + 0.3 * fib + 0.2 * cen
    mix = np.maximum(mix, 1e-6)
    mix = mix / mix.sum()
    chis = np.maximum(chi_min, np.rint(chi_base * mix / mix.mean()).astype(int))
    return chis.tolist()
