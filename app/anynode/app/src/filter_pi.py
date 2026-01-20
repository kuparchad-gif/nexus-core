# src/lilith/metatron/filter_pi.py
import numpy as np
import networkx as nx
from scipy.linalg import eigh
from math import pi as PI, sin, floor

PHI = (1 + 5 ** 0.5) / 2  # golden ratio

def _frac(x: float) -> float:
    return x - np.floor(x)

def _fib(n: int) -> int:
    if n <= 0: return 0
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

class MetatronFilterPI:
    """
    13-node Metatron:
      - Node 0 center; nodes 1..12 ring
      - Edges: 0<->1..12, ring cycle, 3-6-9 chords
    Low-pass in Laplacian eigenspace with π-staggered cutoff:
      cutoff_t = cutoff * (1 + eps * sin(2π t/9 + π/4))
    Horn gain on nodes 0 & 6; optional 0.01 * Gπ(n) shaping.
    """
    def __init__(self, cutoff: float = 0.35, eps: float = 0.08, horn_gain: float = 1.08):
        self.cutoff = float(cutoff)
        self.eps = float(eps)
        self.horn_gain = float(horn_gain)
        self._U = None
        self._lam = None
        self._build_graph_eigens()

    def _build_graph_eigens(self):
        G = nx.Graph()
        G.add_nodes_from(range(13))
        # spokes
        for k in range(1, 13):
            G.add_edge(0, k)
        # ring
        for k in range(1, 13):
            G.add_edge(k, 1 + (k % 12))
        # chords 3-6-9 on ring
        for k in range(1, 13):
            G.add_edge(k, 1 + ((k + 2) % 12))  # +3 modulo 12 (indexing 1..12)
            G.add_edge(k, 1 + ((k + 5) % 12))  # +6
            G.add_edge(k, 1 + ((k + 8) % 12))  # +9

        L = nx.laplacian_matrix(G).astype(float).toarray()
        lam, U = eigh(L)  # symmetric Laplacian ⇒ real eigendecomp
        self._lam = lam
        self._U = U

    def _cutoff_t(self, step: int) -> float:
        t = float(step)
        return self.cutoff * (1.0 + self.eps * np.sin(2.0 * PI * t / 9.0 + PI / 4.0))

    def _G_pi(self, n: int) -> float:
        # Gπ(n) = φ sin(2π/9 (n + frac(nπ))) F_n + ((n + floor(9 frac(nπ))) mod 9) V(n)
        f = _frac(n * PI)
        term1 = PHI * np.sin(2.0 * PI / 9.0 * (n + f)) * _fib(n)
        term2 = ((n + floor(9.0 * f)) % 9) * 1.0  # V(n)=1 baseline
        return float(term1 + term2)

    def apply(self, signal: list[float], step: int) -> list[float]:
        x = np.asarray(signal, dtype=float).reshape(-1)
        if x.shape[0] != 13:
            raise ValueError("MetatronFilterPI expects 13-length signal (nodes 0..12).")

        U, lam = self._U, self._lam
        # Forward eigentransform
        alpha = U.T @ x

        # Low-pass mask
        thr = self._cutoff_t(step)
        mask = (lam <= thr).astype(float)

        # Reconstruct
        y = U @ (alpha * mask)

        # Horn gain on nodes 0 & 6
        y[0] *= self.horn_gain
        y[6] *= self.horn_gain

        # Small π-shaping
        gp = np.array([self._G_pi(i) for i in range(13)], dtype=float)
        y = y + 0.01 * gp

        return y.tolist()
