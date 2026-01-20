
#!/usr/bin/env python3
"""
Builder of Dreams — Smart v2
No switches. You tell it sources and destination. It merges, then fixes.
Largest & newest wins. Then:
 - Profiles the project (Python/Node).
 - Scans Python imports, auto-installs missing (best-effort).
 - Merges Node package.json dependencies (highest version wins), runs npm install if available.
Usage examples:
  - python dream_builder.py
      (then type)   /src/a,/src/b -> /dest
  - python dream_builder.py "/src/a,/src/b -> /dest"
  - python dream_builder.py /src/a /dest
"""
import os, sys, shutil, time, json, subprocess, ast, re
from pathlib import Path

EXCLUDES_DIR = {".git",".hg",".svn",".idea",".vscode","node_modules","dist","build",".venv","venv",".tox","__pycache__"}
EXCLUDES_FILE = {".DS_Store","Thumbs.db"}

BKP_DIRNAME = ".dreams_backups"
REPORT_NAME = "dreams_report.json"

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

def ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def merge_sources(src_paths, dest: Path):
    candidates = {}
    total_scanned = 0
    for s in src_paths:
        if not s.exists(): continue
        base = s
        for p in iter_files(base):
            rel = p.relative_to(base)
            info = file_info(p)
            total_scanned += 1
            candidates.setdefault(str(rel).replace("\\","/"), []).append({
                "path": str(p),
                "size": info["size"],
                "mtime": info["mtime"],
                "origin": f"src:{base}"
            })
    for p in iter_files(dest):
        rel = p.relative_to(dest)
        info = file_info(p)
        candidates.setdefault(str(rel).replace("\\","/"), []).append({
            "path": str(p),
            "size": info["size"],
            "mtime": info["mtime"],
            "origin": "dest"
        })
    applied, skipped, backups, errors = [], [], [], []
    def rank(c): return (c["mtime"], c["size"])
    start = time.time()
    for rel, items in candidates.items():
        items_sorted = sorted(items, key=rank, reverse=True)
        winner = items_sorted[0]
        dest_path = dest / rel
        if dest_path.exists():
            if Path(winner["path"]).resolve() == dest_path.resolve():
                skipped.append({"rel": rel, "reason": "already-best"}); continue
            stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(start))
            bkp = dest / BKP_DIRNAME / stamp / rel
            ensure_parent(bkp)
            try:
                shutil.copy2(dest_path, bkp); backups.append(str(bkp))
            except Exception as e:
                errors.append({"rel": rel, "error": f"backup_failed: {e}"})
        ensure_parent(dest_path)
        try:
            shutil.copy2(winner["path"], dest_path)
            applied.append({"rel": rel, "from": winner["path"], "size": winner["size"], "mtime": winner["mtime"]})
        except Exception as e:
            errors.append({"rel": rel, "error": f"copy_failed: {e}", "from": winner["path"]})
    return {"applied": applied, "skipped": skipped, "backups": backups, "errors": errors, "scanned": total_scanned}

# ---- Smart stuff ----
def profile_project(dest: Path):
    prof = {
        "has_requirements": (dest / "requirements.txt").exists() or (dest / "pyproject.toml").exists(),
        "python_files": [str(p) for p in dest.rglob("*.py") if all(part not in EXCLUDES_DIR for part in p.parts)],
        "has_package_json": (dest / "package.json").exists(),
        "node_dirs": [],
    }
    for p in dest.rglob("package.json"):
        if any(part in EXCLUDES_DIR for part in p.parts): continue
        prof["node_dirs"].append(str(p.parent))
    return prof

def is_stdlib(mod):
    try:
        import sys, importlib.util, importlib.machinery
        if mod in sys.builtin_module_names: return True
        spec = importlib.util.find_spec(mod)
        if spec is None: return False
        # crude check: stdlib usually under .../lib/pythonX.Y/
        return ("python" in (spec.origin or "").lower() and "site-packages" not in (spec.origin or "").lower())
    except Exception:
        return False

def scan_python_missing_modules(py_files):
    missing = set()
    for f in py_files:
        try:
            with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                tree = ast.parse(fh.read(), filename=f)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for n in node.names:
                        root = (n.name.split(".")[0]).strip()
                        if not root or is_stdlib(root): continue
                        try: __import__(root)
                        except Exception: missing.add(root)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        root = node.module.split(".")[0].strip()
                        if not root or is_stdlib(root): continue
                        try: __import__(root)
                        except Exception: missing.add(root)
        except Exception:
            continue
    return sorted(missing)

def pip_install(modules):
    installed, failed = [], []
    for m in modules:
        try:
            print(f"[pip] installing {m} ...")
            out = subprocess.run([sys.executable, "-m", "pip", "install", m], capture_output=True, text=True, timeout=600)
            if out.returncode == 0:
                installed.append(m)
            else:
                failed.append({"module": m, "error": out.stderr[:4000]})
        except Exception as e:
            failed.append({"module": m, "error": str(e)})
    return installed, failed

_semver_num = re.compile(r"(\d+)\.(\d+)\.(\d+)")
def pick_higher_version(a, b):
    # naive: strip ^~>=< and compare numeric triples if present; fallback to a if unknown
    def norm(v):
        v = v.strip()
        v = v.lstrip("^~>=<=")
        m = _semver_num.search(v)
        return tuple(int(x) for x in m.groups()) if m else (0,0,0)
    return a if norm(a) >= norm(b) else b

def merge_package_jsons(dest: Path, source_dirs):
    import json
    pkg_path = dest / "package.json"
    if not pkg_path.exists(): return {"merged": False, "reason": "no_package_json"}
    with open(pkg_path, "r", encoding="utf-8") as f:
        base = json.load(f)
    deps = base.get("dependencies", {}) or {}
    dev = base.get("devDependencies", {}) or {}
    # scan other package.json files in sources
    for s in source_dirs:
        s = Path(s)
        for p in s.rglob("package.json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    other = json.load(f)
                for k,v in (other.get("dependencies") or {}).items():
                    deps[k] = pick_higher_version(v, deps.get(k, v))
                for k,v in (other.get("devDependencies") or {}).items():
                    dev[k] = pick_higher_version(v, dev.get(k, v))
            except Exception:
                continue
    base["dependencies"] = deps
    base["devDependencies"] = dev
    with open(pkg_path, "w", encoding="utf-8") as f:
        json.dump(base, f, indent=2)
    # attempt npm install if npm exists
    npm_ok = shutil.which("npm") is not None
    npm_result = None
    if npm_ok:
        try:
            print("[npm] installing packages ...")
            npm = subprocess.run(["npm","install"], cwd=str(dest), capture_output=True, text=True, timeout=1200)
            npm_result = {"returncode": npm.returncode, "stderr": npm.stderr[-4000:], "stdout": npm.stdout[-2000:]}
        except Exception as e:
            npm_result = {"error": str(e)}
    return {"merged": True, "npm": npm_result}

def main():
    srcs, dest = parse_src_dst(sys.argv[1:])
    dest = Path(dest).resolve()
    src_paths = [Path(s).resolve() for s in srcs]
    for s in src_paths:
        if not s.exists():
            print(f"[WARN] Source not found: {s}", file=sys.stderr)
    if not dest.exists():
        dest.mkdir(parents=True, exist_ok=True)

    # MERGE
    merge_res = merge_sources(src_paths, dest)
    print(f"[MERGE] scanned={merge_res['scanned']} merged={len(merge_res['applied'])} backups={len(merge_res['backups'])} errors={len(merge_res['errors'])}")

    # PROFILE
    prof = profile_project(dest)
    report = {
        "sources": [str(s) for s in src_paths],
        "destination": str(dest),
        "merge": merge_res,
        "profile": prof,
        "python": {},
        "node": {}
    }

    # PYTHON IMPORT FIX
    if prof["python_files"]:
        missing = scan_python_missing_modules(prof["python_files"])
        report["python"]["missing_modules"] = missing
        if missing:
            ins_ok, ins_fail = pip_install(missing)
            report["python"]["installed"] = ins_ok
            report["python"]["install_failed"] = ins_fail

    # NODE MERGE
    if prof["has_package_json"]:
        node_res = merge_package_jsons(dest, src_paths)
        report["node"] = node_res

    # WRITE REPORT
    try:
        with open(dest / REPORT_NAME, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[REPORT] {dest / REPORT_NAME}")
    except Exception as e:
        print(f"[REPORT] failed: {e}", file=sys.stderr)

    print("[DONE]")

if __name__ == "__main__":
    main()
