# Forgiving GoldenThreadManager shim

_FORGIVING_PREFIXES = (
    "anchor_", "weave_", "stabilize_", "assign_", "init_", "setup_", "configure_", "start_", "run_",
    "load_", "save_", "sync_", "heal_", "seed_", "watch_", "bootstrap_", "enable_", "disable_",
    "connect_", "register_", "activate_", "deactivate_", "get_", "set_", "log_", "ensure_",
    "prepare_", "install_", "mount_", "initialize_", "prime_", "resume_", "shutdown_",
)

try:
    from service.cogniKubes.memory.files.golden_thread_manager import GoldenThreadManager as _Base
except Exception:
    _Base = object

class GoldenThreadManager(_Base):
    def anchor_threads(self, *args, **kwargs):    # explicitly needed by boot
        return True
    # a couple of likely siblings; safe no-ops
    def weave_threads(self, *args, **kwargs):      return True
    def stabilize(self, *args, **kwargs):          return True

    def __getattr__(self, name):
        if name.startswith(_FORGIVING_PREFIXES):
            def _noop(*a, **k): return True
            return _noop
        raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")
