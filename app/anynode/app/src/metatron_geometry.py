import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigsh

def build_metatron_graph() -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(13))
    for i in range(1, 13):
        G.add_edge(0, i)
    outer = list(range(1, 13))
    n = len(outer)
    for k in range(n):
        a = outer[k]
        b = outer[(k + 1) % n]
        c = outer[(k + 6) % n]
        G.add_edge(a, b)
        G.add_edge(a, c)
    return G

def laplacian_eig(G: nx.Graph, k: int = 12):
    L = nx.laplacian_matrix(G).astype(float)
    vals, vecs = eigsh(L, k=min(k, L.shape[0]-1), which="SM")
    return vals, vecs

def spectral_ordering_13() -> list:
    G = build_metatron_graph()
    vals, vecs = laplacian_eig(G, k=3)
    fiedler = vecs[:, 1]
    second  = vecs[:, 2] if vecs.shape[1] > 2 else np.zeros_like(fiedler)
    order = sorted(range(len(fiedler)), key=lambda i: (fiedler[i], second[i]))
    return order

def make_permutation(dim: int, base_order: list):
    base = np.array(base_order, dtype=int)
    reps = dim // len(base)
    rem  = dim % len(base)
    tiled = np.concatenate([base for _ in range(reps)] + [base[:rem]]) if reps or rem else base.copy()
    idx = np.arange(dim, dtype=int)
    perm = idx[np.argsort(tiled, kind="mergesort")]
    return perm
