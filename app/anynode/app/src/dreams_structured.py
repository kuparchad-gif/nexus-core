
#!/usr/bin/env python3
"""
Builder of Dreams — Structured v3
Zero-flags cleanup + merge into YOUR canonical layout.

You provide: sources and a destination that represents your CANONICAL structure.
The tool LEARNS that structure (or reads DEST/.dreams/layout.yaml if present),
then ingests any sources, cleans them, and places files where they belong.

- No switches.
- Always: newest mtime wins; tie-breaker: largest size.
- Backups of replaced files to DEST/.dreams_backups/<stamp>/...
- A full move/merge report at DEST/dreams_structured_report.json

Usage:
  python dreams_structured.py
    (type)   /src/a,/src/b -> /dest

  python dreams_structured.py "/src/a,/src/b -> /dest"
  python dreams_structured.py /src/a /dest
"""
import os, sys, shutil, time, json, hashlib, re
from pathlib import Path

EXCLUDES_DIR = {".git",".hg",".svn",".idea",".vscode","node_modules","dist","build",".venv","venv",".tox","__pycache__"}
EXCLUDES_FILE = {".DS_Store","Thumbs.db"}
BKP_DIRNAME = ".dreams_backups"
REPORT_NAME = "dreams_structured_report.json"

# ---------- Input parsing (no flags) ----------
def parse_src_dst(argv):
    if len(argv) == 0:
        line = input("Source(s) -> Destination: ").strip()
    elif len(argv) == 1:
        line = argv[0].strip()
        if "->" not in line:
            return [line], os.getcwd()
    else:
        return [argv[0].strip()], argv[1].strip()

    if "->" not in line:
        return [p.strip() for p in line.split(",") if p.strip()], os.getcwd()
    left, right = line.split("->", 1)
    srcs = [p.strip() for p in left.split(",") if p.strip()]
    dest = right.strip()
    if not srcs or not dest:
        print("Please provide 'SRC[,SRC2...] -> DEST'", file=sys.stderr)
        sys.exit(2)
    return srcs, dest

# ---------- Helpers ----------
def iter_files(base: Path):
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in EXCLUDES_DIR and not d.startswith(".git")]
        for f in files:
            if f in EXCLUDES_FILE: continue
            p = Path(root) / f
            try:
                if p.is_symlink() or not p.is_file(): continue
            except Exception:
                continue
            yield p

def file_info(p: Path):
    try:
        st = p.stat()
        return {"size": st.st_size, "mtime": st.st_mtime}
    except Exception:
        return {"size": -1, "mtime": -1}

def sha256(path: Path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024*1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None

def ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

# ---------- Learn or read layout ----------
DEFAULT_LAYOUT = {
    "apps": ["apps","services","packages","modules"],
    "libs": ["libs","lib"],
    "infra": ["infra","ops","deploy","k8s","helm","terraform"],
    "config": ["config","configs",".config",".configs"],
    "scripts": ["scripts","bin"],
    "docs": ["docs"],
    "tests": ["tests","__tests__"]
}

def load_layout_file(dest: Path):
    # If DEST/.dreams/layout.yaml exists, we use it.
    layout_file = dest / ".dreams" / "layout.yaml"
    if layout_file.exists():
        try:
            import yaml
            with open(layout_file, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            pass
    return None

def learn_layout(dest: Path):
    """Map categories -> chosen canonical folder in DEST.
       If none found, we create them lazily on first use.
    """
    picked = {}
    tops = [p for p in dest.iterdir() if p.is_dir() and p.name not in EXCLUDES_DIR]
    names = {p.name.lower(): p for p in tops}
    for cat, options in DEFAULT_LAYOUT.items():
        chosen = None
        for opt in options:
            if opt.lower() in names:
                chosen = names[opt.lower()]
                break
        picked[cat] = str(chosen if chosen else (dest / opt).as_posix())
    return picked

def categorize(path: Path):
    """Return (category, service) based on file type and nearby project markers"""
    name = path.name.lower()
    parts = [p.lower() for p in path.parts]
    ext = path.suffix.lower()
    # detect service root by markers
    service = None
    for parent in [path.parent] + list(path.parents):
        if (parent / "package.json").exists() or (parent / "pyproject.toml").exists() or (parent / "requirements.txt").exists() or (parent / "go.mod").exists():
            service = parent.name
            break
    # category by path hints
    if "test" in parts or name.startswith("test_") or name.endswith("_test.py") or name.endswith(".spec.js") or name.endswith(".test.ts"):
        cat = "tests"
    elif ext in (".md",".rst",".adoc") or "docs" in parts:
        cat = "docs"
    elif ext in (".yaml",".yml",".json",".toml",".ini",".env",".cfg",".conf"):
        cat = "config"
    elif ext in (".sh",".ps1",".bat",".cmd",".makefile",".mk") or "scripts" in parts or "bin" in parts:
        cat = "scripts"
    elif ext in (".tf",".tfvars",".hcl") or "helm" in parts or "k8s" in parts or "infra" in parts or "deploy" in parts:
        cat = "infra"
    elif ext in (".py",".js",".ts",".tsx",".go",".java",".cs",".rb",".php",".rs",".cpp",".c",".swift",".kt",".m",".mm",".scala",".ex",".exs"):
        # code: if under service, classify as apps; else libs
        cat = "apps" if service else "libs"
    else:
        # default bucket
        cat = "libs" if service is None else "apps"
    # derive service for code; otherwise keep None
    return cat, service

# ---------- Planner ----------
def plan_moves(src_paths, dest: Path, layout_map):
    """Return list of (src_path, dest_path) based on layout and categories"""
    moves = []
    seen = set()
    for src in src_paths:
        if not src.exists(): continue
        for p in iter_files(src):
            # flatten duplicate repo wrappers like a/b/a/b/file -> drop repeated segments if detected
            cat, service = categorize(p)
            base_target = Path(layout_map.get(cat, (dest / cat).as_posix()))
            if service:
                target_dir = Path(base_target) / service
            else:
                # derive a grouping name from nearest package dir name or src root name
                group = src.name
                target_dir = Path(base_target) / group
            rel_within = None
            # preserve relative path from detected service root if any
            if service:
                # find the root dir named service
                root = None
                for parent in [p.parent] + list(p.parents):
                    if parent.name == service:
                        root = parent; break
                if root:
                    try:
                        rel_within = p.relative_to(root)
                    except Exception:
                        rel_within = p.name
                else:
                    rel_within = p.name
            else:
                # if src contains obvious "src" or "lib" nesting, trim one leading layer
                parts = list(p.relative_to(src).parts)
                if parts and parts[0].lower() in ("src","source","lib","app","apps","services"):
                    parts = parts[1:]
                rel_within = Path(*parts) if parts else Path(p.name)
            dst = target_dir / rel_within
            key = (str(p), str(dst))
            if key not in seen:
                moves.append((p, dst))
                seen.add(key)
    return moves

# ---------- Apply with conflict policy ----------
def apply_moves(moves, dest: Path):
    applied = []
    skipped = []
    backups = []
    errors = []
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    for src, dst in moves:
        try:
            ensure_parent(dst)
            if dst.exists():
                # conflict: newest/size wins; if equal and same hash, skip
                src_info = file_info(src); dst_info = file_info(dst)
                if src_info["mtime"] == dst_info["mtime"] and src_info["size"] == dst_info["size"]:
                    # check hash if cheap
                    shs = sha256(src); shd = sha256(dst)
                    if shs and shd and shs == shd:
                        skipped.append({"dst": str(dst), "reason":"identical"})
                        continue
                winner = "src" if (src_info["mtime"], src_info["size"]) > (dst_info["mtime"], dst_info["size"]) else "dst"
                if winner == "dst":
                    skipped.append({"dst": str(dst), "reason":"newer-or-larger-exists"})
                    continue
                # backup existing
                bkp = dest / BKP_DIRNAME / stamp / dst.relative_to(dest)
                ensure_parent(bkp)
                try:
                    shutil.copy2(dst, bkp); backups.append(str(bkp))
                except Exception as e:
                    errors.append({"dst": str(dst), "error": f"backup_failed: {e}"})
            shutil.copy2(src, dst)
            applied.append({"src": str(src), "dst": str(dst)})
        except Exception as e:
            errors.append({"src": str(src), "dst": str(dst), "error": str(e)})
    return applied, skipped, backups, errors

def main():
    srcs, dest_s = parse_src_dst(sys.argv[1:])
    dest = Path(dest_s).resolve()
    src_paths = [Path(s).resolve() for s in srcs]
    for s in src_paths:
        if not s.exists():
            print(f"[WARN] Source not found: {s}", file=sys.stderr)
    dest.mkdir(parents=True, exist_ok=True)

    # load or learn layout
    layout = load_layout_file(dest) or learn_layout(dest)

    # plan moves
    moves = plan_moves(src_paths, dest, layout)

    # apply with conflict policy
    applied, skipped, backups, errors = apply_moves(moves, dest)

    report = {
        "destination": str(dest),
        "layout_used": layout,
        "sources": [str(x) for x in src_paths],
        "counts": {"planned": len(moves), "moved": len(applied), "skipped": len(skipped), "backups": len(backups), "errors": len(errors)},
        "skipped": skipped[:200],
        "errors": errors[:200]
    }
    with open(dest / REPORT_NAME, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[DONE] planned={len(moves)} moved={len(applied)} skipped={len(skipped)} backups={len(backups)} errors={len(errors)}")
    print(f"[REPORT] {dest / REPORT_NAME}")

if __name__ == "__main__":
    main()
