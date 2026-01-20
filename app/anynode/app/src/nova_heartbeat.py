"""
Shim for: from systems.engine.api.nova_heartbeat import Heartbeat
Falls back to a forgiving stub if the uppercase module isn't available.
"""
try:
    from Systems.engine.api.nova_heartbeat import Heartbeat as _Real  # type: ignore
    Heartbeat = _Real
except Exception:
    class Heartbeat:
        def __init__(self, *a, **k): pass
        def start(self, *a, **k): return True
        def stop(self, *a, **k): return True
        def beat(self, *a, **k): return True
        def __getattr__(self, name):
            if name.startswith(("beat", "start", "stop", "send_", "init_", "run_")):
                def _noop(*a, **k): return True
                return _noop
            raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")
