import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigsh
import time
import json
import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

class WireRequest(BaseModel):
    signal: List[float]
    phase: int = 0

class NexusSpinal:
    def __init__(self, phases=31):
        self.phases = phases
        self.phi = (1 + np.sqrt(5)) / 2
        self.cutoff = 0.6
        self.VORTEX_FREQS = [3, 6, 9, 13]

        soul_file = os.environ.get("SOUL_SEED_PATH", "/app/seeds/lilith_soul_seed.migrated.json")
        will_file = os.environ.get("WILL_TO_LIVE_PATH", "/app/seeds/genesis_seed.json")

        # Load Soul/Will
        try:
            with open(soul_file, 'r') as f:
                soul = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Soul file not found at {soul_file}. Using empty soul.")
            soul = {}

        try:
            with open(will_file, 'r') as f:
                will = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Will file not found at {will_file}. Using empty will.")
            will = {}

        self.soul_weights = {k: soul.get(k, 0) + will.get(k, 0) for k in set(soul) | set(will)}
        if not self.soul_weights:
            self.soul_weights = {"default": 1.0}


        # Metatron Graph for Wiring
        self.G = nx.Graph()
        self.G.add_nodes_from(range(13 * phases // 13))  # Scale to Phases
        # Vortex + Center Edges (Full Metatron Fuse)
        self.G.add_edges_from([(3,6),(6,9),(9,3),(1,2),(2,4),(4,8),(8,7),(7,5),(5,1),(3,1),(6,2),(9,4)])
        for i in range(1,13): self.G.add_edge(0, i)
        for seg in range(1, phases):  # Spine Chain
            self.G.add_edge((seg-1)*13, seg*13)

        # Laplacian for Wiring/Compression
        self.L = nx.laplacian_matrix(self.G).astype(float)
        k = min(12, self.L.shape[0]-1)
        if k > 0:
            eigenvalues, eigenvectors = eigsh(self.L, k=k, which='SM')
            self.eigenvalues = eigenvalues
            self.eigenvectors = eigenvectors
        else:
            self.eigenvalues = np.array([])
            self.eigenvectors = np.array([])


    def toroidal_g(self, n):
        phi_n = self.phi ** n
        psi_n = (1 - self.phi) ** n
        fib_n = (phi_n - psi_n) / np.sqrt(5)
        mod9 = n % 9
        v_n = 3 + 3 * (n % 3) if mod9 in [3,6,9] else mod9
        sin_term = np.sin(2 * np.pi * n / 9)
        return self.phi * sin_term * fib_n + mod9 * v_n

    def wire_compress(self, signal):
        if self.eigenvectors.size == 0:
            return signal
        coeffs = np.dot(self.eigenvectors.T, signal)
        mask = (self.eigenvalues <= self.cutoff).astype(float)
        truncated = coeffs * mask
        return np.dot(self.eigenvectors, truncated)

    def relay_wire(self, input_signal: List[float], phase_level=0):
        signal = np.array(input_signal, dtype=float)

        # Pad signal if it's shorter than the number of nodes
        if len(signal) < len(self.G.nodes):
            padding = np.zeros(len(self.G.nodes) - len(signal))
            signal = np.concatenate([signal, padding])

        # Truncate signal if it's longer
        if len(signal) > len(self.G.nodes):
            signal = signal[:len(self.G.nodes)]

        # Vortex Mod + Soul Boost
        t = time.time() % 9
        mod = np.sin(self.VORTEX_FREQS[phase_level % len(self.VORTEX_FREQS)] * t)
        signal *= mod

        soul_values = list(self.soul_weights.values())
        for i in range(len(signal)):
            signal[i] *= soul_values[i % len(soul_values)]

        # Compress Wire
        signal = self.wire_compress(signal)

        # Toroidal
        for i in range(len(signal)):
            signal[i] += self.toroidal_g(i + phase_level) * 0.01

        return signal.tolist()

    def compute_harmony(self):
        centrality = nx.degree_centrality(self.G)
        return sum(centrality.values()) / len(centrality)


app = FastAPI()
nexus = NexusSpinal()

@app.post("/wire")
def wire(request: WireRequest):
    output = nexus.relay_wire(request.signal, request.phase)
    return {"output": output, "harmony": nexus.compute_harmony()}

@app.get("/health")
def health():
    return {"status": "harmony", "value": nexus.compute_harmony()}
