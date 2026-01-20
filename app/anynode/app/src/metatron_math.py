# metatron_math.py — Engineering-grade math primitives for Metatron Engine/Filter
# - Vortex (3-6-9), Fibonacci kernel, Golden-ratio scaling
# - Graph Laplacian spectral filter w/ cutoff + horn boosts
# - Toroidal field generator G(n) per provided formulation
# - Shannon capacity helper
# - Optional SciPy; graceful fallback if eigensolver missing

from __future__ import annotations
import math, time, os
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple, List

import numpy as np

_HAS_NX = _HAS_SCIPY = False
try:
    import networkx as nx
    _HAS_NX = True
except Exception:
    nx = None  # type: ignore
try:
    from scipy.sparse.linalg import eigsh
    _HAS_SCIPY = True
except Exception:
    pass

# ------------------ Core sequences ------------------

def golden_ratio() -> float:
    return (1.0 + math.sqrt(5.0)) / 2.0

def fibonacci_numbers(n: int) -> List[int]:
    """Return first n Fibonacci numbers, F0..F(n-1)."""
    a,b = 0,1
    out = []
    for _ in range(n):
        out.append(a)
        a,b = b,a+b
    return out

def fibonacci_kernel_13(normalize: bool = True) -> np.ndarray:
    """Symmetric 13-length kernel: [1,1,2,3,5,8,13,8,5,3,2,1,1]."""
    arr = np.array([1,1,2,3,5,8,13,8,5,3,2,1,1], dtype=np.float32)
    if normalize:
        s = float(arr.sum())
        if s > 0: arr /= s
    return arr

def digital_root_mod9(n: int) -> int:
    """1..9 cycle; 9 stays 9 (not 0)."""
    if n <= 0: return 9 if n % 9 == 0 else (n % 9)
    r = n % 9
    return 9 if r == 0 else r

def vortex_loop_sequence(length: int) -> List[int]:
    """Repeat 1→2→4→8→7→5 cycle."""
    base = [1,2,4,8,7,5]
    out = []
    while len(out) < length:
        out.extend(base)
    return out[:length]

# ------------------ Toroidal field generator ------------------

def fib_n(n: int) -> float:
    """F_n via fast doubling to avoid large Binet error."""
    def _fd(k: int) -> Tuple[int,int]:
        if k == 0: return (0,1)
        a,b = _fd(k>>1)
        c = a*(2*b - a)
        d = a*a + b*b
        if k & 1:
            return (d, c+d)
        else:
            return (c, d)
    return float(_fd(max(0,int(n)))[0])

def V_of_n(n: int) -> int:
    """V(n) = 3 + 3*(n mod 3) when n mod 9 in {3,6,9}, otherwise digital-root loop value."""
    mod9 = digital_root_mod9(n)
    if mod9 in (3,6,9):
        return 3 + 3 * (n % 3)
    # map to 1-2-4-8-7-5 cycle
    cycle = [1,2,4,8,7,5]
    # use position based on mod9 -> index mapping (deterministic but arbitrary)
    index = (mod9 - 1) % len(cycle)
    return cycle[index]

def toroidal_G(n: int, phi: Optional[float] = None) -> float:
    """G(n) = phi * sin(2π n / 9) * F_n + (n mod 9) * V(n)."""
    ph = phi or golden_ratio()
    return ph * math.sin(2.0*math.pi*n/9.0) * fib_n(n) + (n % 9) * float(V_of_n(n))

# ------------------ Graph construction helpers ------------------

def metatron_block_13() -> "nx.Graph":
    """13-node block: node 0 center + 1..12 ring, with extra 3-6-9 and 1-2-4-8-7-5 links."""
    if not _HAS_NX:
        raise RuntimeError("networkx not available")
    G = nx.Graph()
    G.add_nodes_from(range(13))
    # center star
    for i in range(1,13):
        G.add_edge(0,i)
    # simple ring around
    for i in range(1,13):
        G.add_edge(i, 1 + (i % 12))
    # vortex triangle 3-6-9 (approximate indices on ring)
    tri = [3,6,9]
    G.add_edges_from([(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])])
    # 1-2-4-8-7-5 cycle (indices on ring)
    cyc = [1,2,4,8,7,5]
    for i in range(len(cyc)):
        G.add_edge(cyc[i], cyc[(i+1) % len(cyc)])
    return G

def metatron_spine(phases: int) -> "nx.Graph":
    """Chain phases of 13-node blocks via their centers (0, 13, 26, ...)."""
    if not _HAS_NX:
        raise RuntimeError("networkx not available")
    G = nx.Graph()
    # create disjoint blocks and relabel nodes into global index space
    for seg in range(phases):
        base = metatron_block_13()
        mapping = {i: seg*13 + i for i in range(13)}
        base = nx.relabel_nodes(base, mapping)
        G.update(base)
        if seg > 0:
            G.add_edge((seg-1)*13, seg*13)
    return G

# ------------------ Spectral filter ------------------

@dataclass
class SpectralConfig:
    cutoff: float = 0.6
    phi_scale: float = golden_ratio()
    horns_boost: Tuple[int,int] = (0,6)    # "dual horn" nodes to boost
    horn_gain: float = 1.2
    vortex_freqs: Tuple[int,int,int,int] = (3,6,9,13)
    time_modulus: int = 9

class MetatronFilter:
    def __init__(self, G: Optional["nx.Graph"]=None, config: Optional[SpectralConfig]=None):
        self.G = G if G is not None else (metatron_block_13() if _HAS_NX else None)
        self.cfg = config or SpectralConfig()
        self._eigen = None  # (values, vectors)

    def _eigendecompose(self):
        if not _HAS_NX or not _HAS_SCIPY or self.G is None:
            self._eigen = None
            return
        L = nx.laplacian_matrix(self.G).astype(float)
        k = max(1, min(12, L.shape[0]-1))
        vals, vecs = eigsh(L, k=k, which='SM')
        self._eigen = (vals, vecs)

    def apply(self, signal: np.ndarray, now: Optional[float]=None) -> np.ndarray:
        """Apply vortex modulation + spectral low-pass + phi scale + horn boosts."""
        if self._eigen is None:
            self._eigendecompose()
        x = signal.astype(np.float32).copy()

        # modulation
        t = (now if now is not None else time.time()) % float(self.cfg.time_modulus)
        freq = self.cfg.vortex_freqs[int(t) % len(self.cfg.vortex_freqs)]
        mod = math.sin(float(freq) * t)
        x *= mod

        # low-pass in GFT
        if self._eigen is not None:
            vals, vecs = self._eigen
            coeffs = vecs.T @ x[:vecs.shape[0]]
            mask = (vals <= self.cfg.cutoff).astype(np.float32)
            coeffs = coeffs * mask
            x[:vecs.shape[0]] = (vecs @ coeffs)

        # golden ratio gain
        x *= float(self.cfg.phi_scale)

        # horn boosts
        h0, h1 = self.cfg.horns_boost
        if 0 <= h0 < x.shape[0]: x[h0] *= self.cfg.horn_gain
        if 0 <= h1 < x.shape[0]: x[h1] *= self.cfg.horn_gain
        return x

# ------------------ Information theory ------------------

def shannon_capacity(bandwidth_hz: float, snr_db: float, efficiency: float=0.9) -> float:
    """C = B * log2(1 + SNR) scaled by efficiency. Returns bits/s."""
    snr_linear = 10.0 ** (snr_db / 10.0)
    return float(bandwidth_hz * math.log2(1.0 + snr_linear) * efficiency)

# ------------------ Convenience pipeline ------------------

@dataclass
class PipelineConfig:
    n: int = 13                 # expected signal length for a single block
    normalize: bool = True
    use_fib_kernel: bool = True
    spectral: SpectralConfig = SpectralConfig()

class MetatronPipeline:
    """One-stop pipeline using the math elements + Laplacian filter."""
    def __init__(self, G: Optional["nx.Graph"]=None, cfg: Optional[PipelineConfig]=None):
        self.cfg = cfg or PipelineConfig()
        self.filter = MetatronFilter(G=G, config=self.cfg.spectral)

    def run(self, signal: np.ndarray, phase_offset: int=0) -> np.ndarray:
        x = signal.astype(np.float32).copy()
        # optional Fibonacci convolution (circular) for 13-length
        if self.cfg.use_fib_kernel and x.shape[0] >= 13:
            k = fibonacci_kernel_13(normalize=True)
            # circular conv same-length
            x = np.real(np.fft.ifft(np.fft.fft(x) * np.fft.fft(k, n=x.shape[0]))).astype(np.float32)

        # spectral filter + vortex modulation + horns + phi
        x = self.filter.apply(x)

        # toroidal post-modulation per index
        tor = np.array([toroidal_G(i+phase_offset) for i in range(x.shape[0])], dtype=np.float32)
        x = x * tor

        if self.cfg.normalize:
            nrm = float(np.linalg.norm(x))
            if nrm > 0: x = x / nrm
        return x

# ------------------ Quick self-test ------------------
if __name__ == "__main__":
    sig = np.random.rand(13).astype(np.float32)
    if _HAS_NX:
        G = metatron_block_13()
    else:
        G = None
    pipe = MetatronPipeline(G=G)
    y = pipe.run(sig, phase_offset=13)
    print("in :", np.round(sig, 3))
    print("out:", np.round(y, 3))
    # capacity examples
    print("C_sound(13*4kHz, 20dB):", shannon_capacity(13*4000, 20.0))
    print("C_light(100kHz, 30dB):", shannon_capacity(100000, 30.0))
