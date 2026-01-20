
import os, re, json, sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
INCLUDE_EXT = {".env", ".env.example", ".yaml", ".yml", ".json", ".toml", ".py", ".ts", ".js", ".tsx", ".jsx", ".go", ".rs", ".java", ".cs", ".tf", ".ini", ".cfg", ".conf", ""}
ENV_RE = re.compile(r'\b([A-Z][A-Z0-9_]{2,})\b')
DOCKER_ENV_RE = re.compile(r'ENV\s+([A-Z][A-Z0-9_]+)\s*=|\bARG\s+([A-Z][A-Z0-9_]+)')

services = {}
env_keys = set()

def scan_file(p: Path):
    try:
        txt = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return
    # collect env-like tokens
    for m in ENV_RE.finditer(txt):
        k = m.group(1)
        # Filter common non-env tokens
        if k in {"HTTP","GET","POST","TRUE","FALSE","JSON","PORT","HOST","NAME","URL"}:
            continue
        env_keys.add(k)
    # crude service detection
    if p.name.lower() in {"dockerfile","containerfile"}:
        srv = p.parent.name
        services.setdefault(srv, {"path": str(p.parent), "dockerfile": str(p)})
        for m in DOCKER_ENV_RE.finditer(txt):
            k = m.group(1) or m.group(2)
            if k: env_keys.add(k)
    elif p.name == "package.json":
        srv = p.parent.name
        services.setdefault(srv, {"path": str(p.parent)})
    elif p.name in {"pyproject.toml","requirements.txt","setup.cfg"}:
        srv = p.parent.name
        services.setdefault(srv, {"path": str(p.parent)})
    elif p.name == "compose.yaml" or p.name == "docker-compose.yml":
        services.setdefault("compose", {"path": str(p.parent), "compose": str(p)})

def should_scan(p: Path):
    if p.is_dir():
        bn = p.name.lower()
        if bn in {".git",".idea",".vscode","node_modules","dist","build","__pycache__",".venv","venv",".mypy_cache",".pytest_cache"}:
            return False
        return True
    else:
        ext = p.suffix.lower()
        if p.name.lower() in {"dockerfile","containerfile"}:
            return True
        return (ext in INCLUDE_EXT) or any(p.name.lower().endswith(s) for s in [".env", ".yml", ".yaml"])

for dp, dn, fn in os.walk(ROOT):
    d = Path(dp)
    if not should_scan(d):
        dn[:] = []  # stop recursion
        continue
    for f in fn:
        p = d / f
        if should_scan(p):
            scan_file(p)

out = {
    "root": str(ROOT),
    "env_keys": sorted(env_keys),
    "services": services
}
print(json.dumps(out, indent=2))
