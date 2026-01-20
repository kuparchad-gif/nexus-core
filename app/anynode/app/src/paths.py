from __future__ import annotations
import os
from pathlib import Path

# Repo root resolution:
# 1) LILLITH_HOME if set
# 2) Walk up from this file until we find a folder containing "src"
def repo_root() -> Path:
    if os.environ.get("LILLITH_HOME"):
        return Path(os.environ["LILLITH_HOME"]).expanduser().resolve()
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "src").is_dir():
            return p
    return Path.cwd()

def resolve_path(posix_like: str | None) -> Path | None:
    """Resolve a POSIX-like relative/absolute path in a portable way under repo root."""
    if not posix_like:
        return None
    p = Path(posix_like).expanduser()
    if not p.is_absolute():
        p = repo_root() / p
    return p.resolve()

def local_first(model_path: str | None, env_override: str = "LILLITH_LOCAL_MODEL_PATH") -> Path | None:
    """Resolve local model path (env overrides config). Return Path if exists, else None."""
    override = os.environ.get(env_override)
    if override:
        po = Path(override).expanduser().resolve()
        if po.exists():
            return po
    pr = resolve_path(model_path)
    return pr if (pr and pr.exists()) else None

def is_hf_id(s: str | None) -> bool:
    """Heuristic: treat as HF repo id if it doesn't exist as a local path and contains '/'."""
    if not s:
        return False
    return "/" in s and not Path(s).exists()
