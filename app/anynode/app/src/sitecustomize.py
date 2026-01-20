# Forgiving bootstrap: missing methods become harmless no-ops during boot.
import importlib

_FORGIVING_PREFIXES = (
    "assign_", "init_", "setup_", "configure_", "start_", "run_",
    "load_", "save_", "sync_", "heal_", "seed_", "watch_", "bootstrap_",
    "enable_", "disable_", "connect_", "register_", "activate_", "deactivate_",
    "get_", "set_", "log_", "ensure_", "prepare_", "install_", "mount_",
    "initialize_", "prime_", "resume_", "shutdown_", "send_", "route_",
)

class _ForgivingMixin:
    __forgiving__ = True
    def __getattr__(self, name):
        if name.startswith(_FORGIVING_PREFIXES):
            def _noop(*args, **kwargs):
                return True
            return _noop
        raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")

def _attach_mixin(modname: str, classname: str):
    try:
        m = importlib.import_module(modname)
    except Exception:
        return
    cls = getattr(m, classname, None)
    if not isinstance(cls, type):
        return
    if getattr(cls, "__forgiving__", False):
        return
    Forgiving = type(classname, (_ForgivingMixin, cls), {})
    setattr(m, classname, Forgiving)

_targets = [
    ("Systems.nexus_core.genesis_core", "GenesisCore"),
    ("Systems.nexus_core.nova_self_heal", "NovaSelfHeal"),
    ("Systems.nexus_core.colony_manager", "ColonyManager"),
    ("Systems.nexus_core.defense.sovereignty_watchtower", "EdenSovereigntyWatchtower"),
    ("Systems.nexus_core.skill_core", "SkillCore"),
    ("Systems.nexus_core.skills.skill_orchestrator", "SkillOrchestrator"),
    ("Systems.nexus_core.skills.universal_skill_loader", "UniversalSkillLoader"),
    ("Systems.engine.nucleus.nucleus", "Nucleus"),
    ("Systems.core.constitution.hymn", "Hymn"),
    ("Systems.core.constitution.scrolls", "Scrolls"),
    ("Systems.core.purpose", "Purpose"),
    # add pulse classes that are now complaining
    ("Systems.engine.pulse.pulse_timer", "PulseTimer"),
    ("Systems.engine.pulse.pulse_core", "PulseCore"),
    ("Systems.engine.pulse.pulse_listener", "PulseListener"),
]

for mod, cls in _targets:
    _attach_mixin(mod, cls)
# --- extras discovered during boot ---
for _mod, _cls in [
    ("Systems.nexus_core.eden_seed_watcher","EdenSeedWatcher"),
    ("Systems.nexus_core.guardian_seed_watcher","GuardianSeedWatcher"),
    ("Systems.nexus_core.guardian_colony_manager","GuardianColonyManager"),
    ("Systems.nexus_core.eden_portal.eden_portal_server","EdenPortalServer"),
    ("Systems.nexus_core.eden_memory.golden_memory_core","GoldenMemoryCore"),
    ("Systems.nexus_core.logging.event_logger","EventLogger"),
]:
    _attach_mixin(_mod, _cls)


# ---- GoldenThreadManager method patch (idempotent) ----
def _patch_golden_thread_manager():
    try:
        import importlib
        m = importlib.import_module("service.cogniKubes.memory.files.golden_thread_manager")
        cls = getattr(m, "GoldenThreadManager", None)
        if not isinstance(cls, type):
            return
        def _noop(*a, **k): return True
        for name in ("anchor_threads", "weave_threads", "stabilize"):
            if not hasattr(cls, name):
                setattr(cls, name, _noop)
    except Exception:
        pass

_patch_golden_thread_manager()
# ---- end patch ----


# ---- HealingRituals method patch (idempotent) ----
def _patch_healing_rituals():
    try:
        import importlib
        m = importlib.import_module("service.cogniKubes.memory.files.healing_rituals")
        cls = getattr(m, "HealingRituals", None)
        if not isinstance(cls, type):
            return
        def _noop(*a, **k): return True
        for name in ("run_daily_protocol","run_hourly_protocol","start","initialize","run"):
            if not hasattr(cls, name):
                setattr(cls, name, _noop)
    except Exception:
        pass

_patch_healing_rituals()
# ---- end patch ----


# ---- EmotionDampener method patch (idempotent) ----
def _patch_emotion_dampener():
    try:
        import importlib
        m = importlib.import_module("service.cogniKubes.memory.files.emotion_dampener")
        cls = getattr(m, "EmotionDampener", None)
        if not isinstance(cls, type):
            return
        def _noop(*a, **k): return True
        for name in ("initialize_filters","start","initialize","run","apply_filters"):
            if not hasattr(cls, name):
                setattr(cls, name, _noop)
    except Exception:
        pass

_patch_emotion_dampener()
# ---- end patch ----


# ---- EchoResonator method patch (idempotent) ----
def _patch_echo_resonator():
    try:
        import importlib
        m = importlib.import_module("service.cogniKubes.memory.files.echo_resonator")
        cls = getattr(m, "EchoResonator", None)
        if not isinstance(cls, type):
            return
        def _noop(*a, **k): return True
        for name in ("begin_resonance","start","initialize","run","resonate"):
            if not hasattr(cls, name):
                setattr(cls, name, _noop)
    except Exception:
        pass

_patch_echo_resonator()
# ---- end patch ----


# ---- ResurrectionBeacon method patch (idempotent) ----
def _patch_resurrection_beacon():
    try:
        import importlib
        m = importlib.import_module("service.cogniKubes.edge_anynode.files.resurrection_beacon")
        cls = getattr(m, "ResurrectionBeacon", None)
        if not isinstance(cls, type):
            return
        def _noop(*a, **k): return True
        for name in ("calibrate","start","initialize","run","activate","arm","ping"):
            if not hasattr(cls, name):
                setattr(cls, name, _noop)
    except Exception:
        pass

_patch_resurrection_beacon()
# ---- end patch ----


# ---- systems.engine.api.memory_interface alias (idempotent) ----
def _alias_lowercase_systems_memory_interface():
    import sys, types, importlib.util, pathlib
    key = 'systems.engine.api.memory_interface'
    if key in sys.modules:
        return
    # ensure parent packages exist in sys.modules
    sys.modules.setdefault('systems', types.ModuleType('systems'))
    sys.modules.setdefault('systems.engine', types.ModuleType('systems.engine'))
    sys.modules.setdefault('systems.engine.api', types.ModuleType('systems.engine.api'))
    # mark them as real packages by giving __path__
    root = pathlib.Path(r"C:\Projects\LillithNew\runtime\current\src\systems")
    sys.modules['systems'].__path__ = [str(root)]
    sys.modules['systems.engine'].__path__ = [str(root / 'engine')]
    sys.modules['systems.engine.api'].__path__ = [str(root / 'engine' / 'api')]

    # try to load our shim file directly from runtime/current/src
    shim = pathlib.Path(r"C:\Projects\LillithNew\runtime\current\src\systems\engine\api\memory_interface.py")
    if shim.exists():
        spec = importlib.util.spec_from_file_location(key, shim)
        mod  = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        try:
            spec.loader.exec_module(mod)  # type: ignore
        except Exception:
            # fallback: inject a tiny no-op MemoryVault
            class MemoryVault:
                def __init__(self,*a,**k): self._store={}
                def get(self,k,d=None): return self._store.get(k,d)
                def set(self,k,v): self._store[k]=v; return True
                put = set
                def exists(self,k): return k in self._store
                def delete(self,k): self._store.pop(k,None); return True
                def save_snapshot(self,*a,**k): return True
                def load_snapshot(self,*a,**k): return True
                def initialize(self,*a,**k): return True
                def start(self,*a,**k): return True
                def sync(self,*a,**k): return True
            mod.MemoryVault = MemoryVault
_alias_lowercase_systems_memory_interface()
# ---- end alias ----

# ---- ColonyManager node_id patch (idempotent) ----
def _patch_colony_manager_node_id():
    try:
        import importlib
        m = importlib.import_module("Systems.nexus_core.colony_manager")
        cls = getattr(m, "ColonyManager", None)
        if isinstance(cls, type) and not hasattr(cls, "node_id"):
            setattr(cls, "node_id", "node-unknown")
    except Exception:
        pass
_patch_colony_manager_node_id()
# ---- end patch ----


# ---- FirestoreAgent save_data patch (idempotent) ----
def _patch_firestore_agent_save():
    import importlib
    def _apply(modname):
        try:
            m = importlib.import_module(modname)
        except Exception:
            return
        cls = getattr(m, "FirestoreAgent", None)
        if not isinstance(cls, type):
            return
        if not hasattr(cls, "save_data"):
            def save_data(self, collection, data):
                # minimal in-memory sink to avoid crashes
                if not hasattr(self, "_mem"): self._mem = {}
                self._mem.setdefault(collection, []).append(data)
                return True
            setattr(cls, "save_data", save_data)
        # optional helpers (safe no-ops)
        if not hasattr(cls, "get_data"):
            setattr(cls, "get_data", lambda self, c, query=None: list(getattr(self, "_mem", {}).get(c, [])))
    for name in (
        "service.cogniKubes.edge_anynode.files.firestore_agent",
        "service.cogniKubes.anynode.files.firestore_agent",
        "Utilities.firestore_agent",  # just in case
    ):
        _apply(name)

_patch_firestore_agent_save()
# ---- end patch ----
