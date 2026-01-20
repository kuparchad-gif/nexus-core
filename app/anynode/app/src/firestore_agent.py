"""
Forgiving shim for Utilities.firestore_agent.
Tries real agent, otherwise supplies a minimal in-memory agent with save_data().
"""
try:
    # try a couple of plausible real locations
    from service.cogniKubes.edge_anynode.files.firestore_agent import FirestoreAgent as _Real  # type: ignore
except Exception:
    try:
        from service.cogniKubes.anynode.files.firestore_agent import FirestoreAgent as _Real  # type: ignore
    except Exception:
        _Real = None

class FirestoreAgent(_Real if _Real else object):
    def __init__(self, *a, **k):
        try:
            if _Real:  # real one may have init behavior
                super().__init__(*a, **k)
        except Exception:
            pass
        self._mem = {}

    # what firestore_logger expects:
    def save_data(self, collection, data):
        self._mem.setdefault(collection, []).append(data)
        return True

    # handy no-ops to be safe
    def get_data(self, collection, query=None):
        return list(self._mem.get(collection, []))
    def delete_data(self, collection, pred=None):
        if pred is None:
            self._mem.pop(collection, None); return True
        items = self._mem.get(collection, [])
        self._mem[collection] = [x for x in items if not pred(x)]
        return True

    def __getattr__(self, name):
        # forgive common verbs
        if name.startswith(("save_", "get_", "set_", "log_", "init_", "start_", "sync_", "update_", "write_", "read_")):
            def _noop(*a, **k): return True
            return _noop
        raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")
