# nexus_brain_metaengine.py — Quantum Meme Brain (optional heavy deps)
# Safe-by-default: all heavy libs are optional. Falls back gracefully.
# - If transformers available, uses bert-base-uncased for embeddings.
# - If qiskit available, uses a 1-qubit H+measure coinflip; else OS randomness.
# - Integrates with NexusSpinal.relay_wire for spinal processing.
import os, time, json, numpy as np

# Optional imports (guarded)
_HAS_TORCH = False
_HAS_TRANSFORMERS = False
_HAS_QISKIT = False

try:
    import torch  # noqa: F401
    _HAS_TORCH = True
    from transformers import BertModel, BertTokenizer
    _HAS_TRANSFORMERS = True
except Exception:
    pass

try:
    from qiskit import QuantumCircuit, Aer, execute  # execute is deprecated in newer qiskit; still widely present
    _HAS_QISKIT = True
except Exception:
    # Newer qiskit may not expose execute; we will fallback to os randomness
    _HAS_QISKIT = False

def _rand_bit() -> int:
    # Try quantum coin. If qiskit not available, use OS randomness.
    if _HAS_QISKIT:
        try:
            backend = Aer.get_backend('qasm_simulator')
            qc = QuantumCircuit(1, 1)
            qc.h(0); qc.measure(0, 0)
            result = execute(qc, backend=backend, shots=1).result()
            counts = result.get_counts()
            return 1 if counts.get('1', 0) >= 1 else 0
        except Exception:
            pass
    # Fallback
    return int.from_bytes(os.urandom(1), 'little') & 1

class QuantumMemeInjector:
    def __init__(self, model_name: str = 'bert-base-uncased'):
        self.available = {"torch": _HAS_TORCH, "transformers": _HAS_TRANSFORMERS, "qiskit": _HAS_QISKIT}
        self.model_name = model_name
        self.model = None
        self.tok = None
        if _HAS_TRANSFORMERS:
            try:
                # Respect offline environments; allow opt-in downloads
                local_only = os.environ.get("BRAIN_ALLOW_HF_DOWNLOADS", "0") != "1"
                self.tok = BertTokenizer.from_pretrained(model_name, local_files_only=local_only)
                self.model = BertModel.from_pretrained(model_name, local_files_only=local_only)
            except Exception:
                self.model = None
                self.tok = None

    def _embed_fallback(self, text: str) -> np.ndarray:
        # Simple deterministic hashing to 768 dims
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        return rng.normal(size=(1, 768)).astype(np.float32)

    def entangle_meme(self, text: str) -> np.ndarray:
        """Encode text to a vector, then quantum-perturb (roll) based on coin flip."""
        if self.model is not None and self.tok is not None:
            try:
                import torch
                with torch.no_grad():
                    inputs = self.tok(text, return_tensors="pt", truncation=True, max_length=256)
                    outputs = self.model(**inputs)
                    emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy().astype(np.float32)  # (1, 768)
            except Exception:
                emb = self._embed_fallback(text)
        else:
            emb = self._embed_fallback(text)

        # "Quantum" randomness: rotate vector by 1 position if bit=1
        if _rand_bit() == 1:
            emb = np.roll(emb, 1, axis=1)
        return emb  # shape (1, 768)

class MetaBrain:
    """Bridges QuantumMemeInjector to the Nexus spinal relay."""
    def __init__(self, spinal_cord, phase_level: int = 13):
        self.spinal_cord = spinal_cord
        self.phase_level = phase_level
        self.meme_injector = QuantumMemeInjector()
        self.stats = {"rewires": 0}

    def think(self, input_text: str):
        vec = self.meme_injector.entangle_meme(input_text)  # (1, D)
        # Optional neuroplastic touch: light rewiring proxy (incremental stat)
        self.stats["rewires"] += 1
        # Send through spinal relay (pads/trim handled by relay)
        out = self.spinal_cord.relay_wire(vec.flatten(), phase_level=self.phase_level)
        # Return a compact summary
        return {
            "vector_sample": [float(x) for x in np.asarray(out).reshape(-1)[:10]],
            "nodes": int(self.spinal_cord.G.number_of_nodes()),
            "phase": int(self.phase_level),
            "deps": self.meme_injector.available,
            "rewires": int(self.stats["rewires"])
        }
