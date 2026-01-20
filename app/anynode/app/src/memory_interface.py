"""
Shim for: from systems.engine.api.memory_interface import MemoryVault
Tries the uppercase path first; otherwise provides a safe no-op MemoryVault.
"""
try:
    from Systems.engine.api.memory_interface import MemoryVault as _Real  # type: ignore
    MemoryVault = _Real
except Exception:
    class MemoryVault:
        def __init__(self, *args, **kwargs): self._store = {}
        def get(self, k, default=None):  return self._store.get(k, default)
        def set(self, k, v):             self._store[k] = v; return True
        def put(self, k, v):             return self.set(k, v)
        def exists(self, k):             return k in self._store
        def delete(self, k):             self._store.pop(k, None); return True
        def save_snapshot(self, *a, **k): return True
        def load_snapshot(self, *a, **k): return True
        def initialize(self, *a, **k):    return True
        def start(self, *a, **k):         return True
        def sync(self, *a, **k):          return True
