# map_and_fix_imports.py
# Scans your repo, finds missing imports under prefixes (Utilities, Systems),
# proposes targets from your real tree, and can (a) report, (b) write shims,
# or (c) rewrite the imports in-place (with .bak backups).
#
# Optional: --lm will ask LM Studio (http://localhost:1234) for confirmation.

from __future__ import annotations

import argparse, json, os, re, sys, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

# -------- settings (defaults) --------
DEFAULT_PREFIXES = ["Utilities", "Systems"]
DEFAULT_SEARCH_ORDER = [
    "service.cogniKubes.heart.files",
    "service.cogniKubes.anynode.files",
    "service",
    "files",
    "utils",
]

# hard exact maps you already hit
DEFAULT_EXACT_MAP = {
    "Systems.engine.api.manifest_server": "service.cogniKubes.anynode.files.manifest_server",
}

CHAT_URL = os.environ.get("LMSTUDIO_CHAT_URL", "http://localhost:1234/v1/chat/completions")
CHAT_MODEL = os.environ.get("LMSTUDIO_MODEL", "local-model")  # LM Studio ignores this and uses the selected model


# -------- indexing --------
def dotted_from(root: Path, file: Path) -> Optional[str]:
    try:
        rel = file.relative_to(root)
    except Exception:
        return None
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1].removesuffix(".py")
    if not parts:
        return None
    return ".".join(parts)


def build_index(src_root: Path) -> Dict[str, Set[str]]:
    """
    Returns:
      {
        "all": set of dotted modules found,
        "by_tail": { 'module_name': { dotted1, dotted2, ... } }
      }
    """
    allmods: Set[str] = set()
    by_tail: Dict[str, Set[str]] = {}
    for p in src_root.rglob("*.py"):
        d = dotted_from(src_root, p)
        if not d:
            continue
        allmods.add(d)
        tail = d.split(".")[-1]
        by_tail.setdefault(tail, set()).add(d)
    return {"all": allmods, "by_tail": by_tail}


# -------- parsing imports --------
IMPORT_RE = re.compile(
    r"""
    (?P<kind>from|import)       # keyword
    \s+
    (?P<mod>[A-Za-z_][\w\.]*)   # module dotted
    """,
    re.VERBOSE,
)


def extract_expected_modules(py_text: str, prefixes: List[str]) -> List[str]:
    found: Set[str] = set()
    for m in IMPORT_RE.finditer(py_text):
        mod = m.group("mod")
        if any(mod == p or mod.startswith(p + ".") for p in prefixes):
            found.add(mod)
    return sorted(found)


# -------- heuristic mapping --------
def rank_candidate(cand: str, search_order: List[str]) -> Tuple[int, str]:
    # lower is better
    for i, p in enumerate(search_order):
        if cand == p or cand.startswith(p + "."):
            return (-100 + i, cand)
    return (0, cand)


def guess_mapping(expected: str, idx: Dict[str, Set[str]], search_order: List[str]) -> Optional[str]:
    tail = expected.split(".")[-1]
    cands = list(idx["by_tail"].get(tail, []))
    if not cands:
        return None
    cands.sort(key=lambda c: rank_candidate(c, search_order))
    return cands[0]


# -------- LM Studio assist (optional) --------
def lm_confirm(expected: str, proposed: Optional[str], context: str = "") -> Optional[str]:
    """
    Ask LM Studio to confirm/adjust the target module. Return string or None.
    """
    try:
        import requests  # std in LM Studio environment usually has it; if not, we just skip
    except Exception:
        return proposed

    prompt = (
        "You are a terse senior Python build engineer.\n"
        f"We are fixing a missing import: `{expected}`.\n"
        f"Proposed mapping target: `{proposed}`.\n"
        "Context (code paths exist in the repo):\n"
        f"{context}\n"
        "Reply with ONLY the final dotted module path to import (or the word NONE)."
    )
    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": "You rewrite imports precisely."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
    }
    try:
        r = requests.post(CHAT_URL, json=payload, timeout=20)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"].strip()
        if content.upper() == "NONE":
            return None
        # sanitize to a dotted module-ish token
        content = content.split()[0].strip("`'\"")
        return content
    except Exception:
        return proposed


# -------- apply fixes --------
def ensure_pkg(dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    init = dest / "__init__.py"
    if not init.exists():
        init.write_text("", encoding="utf-8")


def write_shim(root: Path, expected: str, real: str, overwrite: bool = False):
    """
    Create package tree matching `expected` and write a shim file that re-exports from `real`.
    """
    parts = expected.split(".")
    pkg_parts, leaf = parts[:-1], parts[-1]
    pkg_dir = root.joinpath(*pkg_parts)
    ensure_pkg(pkg_dir)
    shim = pkg_dir / f"{leaf}.py"
    if shim.exists() and not overwrite:
        return
    shim.write_text(
        f"# Autogenerated shim so `{expected}` imports `{real}`\n"
        f"from {real} import *  # noqa: F401,F403\n",
        encoding="utf-8",
    )


def rewrite_imports_in_file(path: Path, mapping: Dict[str, str]) -> bool:
    txt = path.read_text(encoding="utf-8", errors="ignore")
    orig = txt
    for exp, real in mapping.items():
        # Replace: "from exp import" and "import exp"
        txt = re.sub(rf"\bfrom\s+{re.escape(exp)}\s+import\b", f"from {real} import", txt)
        txt = re.sub(rf"\bimport\s+{re.escape(exp)}\b", f"import {real}", txt)
    if txt != orig:
        bak = path.with_suffix(path.suffix + ".bak")
        if not bak.exists():
            bak.write_text(orig, encoding="utf-8")
        path.write_text(txt, encoding="utf-8")
        return True
    return False


# -------- main --------
def main():
    ap = argparse.ArgumentParser(description="Map and fix missing Utilities/* & Systems/* imports.")
    ap.add_argument("--root", required=True, help="Project src root (e.g., C:\\Projects\\LillithNew\\src)")
    ap.add_argument("--mode", choices=["map", "shim", "rewrite"], default="map")
    ap.add_argument("--apply", action="store_true", help="Actually write shims / rewrite files")
    ap.add_argument("--prefix", action="append", default=[], help="Extra import prefixes to include")
    ap.add_argument("--search", action="append", default=[], help="Prepend to search order preference")
    ap.add_argument("--exact", type=str, default="", help="Extra exact map JSON, e.g. {'A.B':'x.y'}")
    ap.add_argument("--force", action="store_true", help="Overwrite shims if they already exist")
    ap.add_argument("--out", default="import_map_report.json", help="Report path (map mode)")
    ap.add_argument("--lm", action="store_true", help="Ask LM Studio to confirm/adjust mapping")
    args = ap.parse_args()

    src_root = Path(args.root).resolve()
    if not src_root.exists():
        print(f"[ERR] root not found: {src_root}", file=sys.stderr)
        sys.exit(2)

    prefixes = DEFAULT_PREFIXES + args.prefix
    search_order = list(args.search) + DEFAULT_SEARCH_ORDER

    exact_map = dict(DEFAULT_EXACT_MAP)
    if args.exact:
        try:
            exact_map.update(json.loads(args.exact.replace("'", '"')))
        except Exception as e:
            print(f"[WARN] --exact parse failed: {e}", file=sys.stderr)

    idx = build_index(src_root)

    # gather imports
    all_expected: Set[str] = set()
    py_files: List[Path] = [p for p in src_root.rglob("*.py")]
    for p in py_files:
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for mod in extract_expected_modules(txt, prefixes):
            all_expected.add(mod)

    # decide mappings
    decisions: Dict[str, Dict[str, str | None]] = {}
    for exp in sorted(all_expected):
        real = None
        if exp in exact_map:
            real = exact_map[exp]
        else:
            real = guess_mapping(exp, idx, search_order)

        if args.lm:
            # small context: list a few candidates by same tail to help the model
            tail = exp.split(".")[-1]
            cands = sorted(idx["by_tail"].get(tail, []))[:12]
            ctx = "\n".join(cands)
            real = lm_confirm(exp, real, ctx)

        decisions[exp] = {
            "real": real,
            "status": "mapped" if real else "unresolved",
        }

    # modes
    if args.mode == "map":
        report = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "root": str(src_root),
            "prefixes": prefixes,
            "search_order": search_order,
            "decisions": decisions,
        }
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[OK] wrote report -> {args.out}")
        # quick summary
        unresolved = [k for k, v in decisions.items() if v["real"] is None]
        if unresolved:
            print("[WARN] unresolved:", ", ".join(unresolved))
        return

    if args.mode == "shim":
        if not args.apply:
            print("[DRY] shim mode: use --apply to write files")
            for exp, meta in decisions.items():
                print(f"  {exp} -> {meta['real']}")
            return
        count = 0
        for exp, meta in decisions.items():
            real = meta["real"]
            if not real:
                print(f"[SKIP] unresolved: {exp}")
                continue
            write_shim(src_root, exp, real, overwrite=args.force)
            count += 1
        print(f"[OK] wrote {count} shims under {src_root}")
        return

    if args.mode == "rewrite":
        mapping = {exp: meta["real"] for exp, meta in decisions.items() if meta["real"]}
        if not args.apply:
            print("[DRY] rewrite mode: use --apply to modify files")
            for exp, real in mapping.items():
                print(f"  {exp} -> {real}")
            return
        touched = 0
        for p in py_files:
            try:
                if rewrite_imports_in_file(p, mapping):
                    touched += 1
                    print(f"[PATCH] {p}")
            except Exception as e:
                print(f"[ERR] {p}: {e}", file=sys.stderr)
        print(f"[OK] rewrote imports in {touched} file(s). Backups: *.py.bak")
        return


if __name__ == "__main__":
    main()
