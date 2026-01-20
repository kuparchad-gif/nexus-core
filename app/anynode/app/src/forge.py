#!/usr/bin/env python3
"""
forge.py — Model-agnostic repo refactor + dataset builder using LM Studio (DeepSeek, Codestral, etc.)

Goals
- Keep /src as the root (cloud-style), remove OS-specific hardcoding
- Scan codebase, ask model for refactors/fixes, MERGE-NOT-OVERWRITE
- Build a manifest and training pairs for student models (AcidemiKubes)

No external deps required (uses stdlib only). Point it at LM Studio (default http://127.0.0.1:1234).

Example
  python forge.py --root /src --list-models
  python forge.py --root /src --model deepseek-coder --apply
  python forge.py --root /src --model codestral --dry-run

Exit code 0 on success; prints a single line starting with 'FORGE COMPLETE' when finished.
"""
from __future__ import annotations
import argparse
import difflib
import fnmatch
import hashlib
import io
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

LMSTUDIO_URL = os.environ.get("LMSTUDIO_URL", "http://127.0.0.1:1234")
CHAT_PATH = "/v1/chat/completions"
MODELS_PATH = "/v1/models"

DEFAULT_EXCLUDES = [
    ".git", "node_modules", ".venv", "venv", "__pycache__", ".forge",
]
DEFAULT_INCLUDE_GLOBS = ["**/*.py", "**/*.json", "**/*.yaml", "**/*.yml", "**/*.toml", "**/*.md"]

SYSTEM_PROMPT = (
    "You are a senior Python build engineer and refactoring assistant.\n"
    "Task: Rewrite the given file to use /src as the root (POSIX style), remove Windows-specific absolute paths,\n"
    "avoid sys.path hacks when possible, and make imports work in cloud environments.\n"
    "\nRules:\n"
    "- Keep behavior.\n"
    "- If file is JSON/YAML/TOML config, normalize paths under /src.\n"
    "- NEVER delete content; if unsure, comment and leave TODO.\n"
    "- Output ONLY the full, revised file content. No explanations.\n"
)

def http_json(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))

def list_models() -> List[str]:
    try:
        res = http_json(LMSTUDIO_URL + MODELS_PATH, {})
        data = res if isinstance(res, dict) else {}
        models = []
        for m in data.get("data", []):
            mid = m.get("id") or m.get("name")
            if mid:
                models.append(mid)
        return models
    except Exception:
        return []

def chat_complete(model: str, messages: List[Dict[str, str]], temperature: float = 0.1, max_tokens: int = 2048) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    res = http_json(LMSTUDIO_URL + CHAT_PATH, payload)
    try:
        return res["choices"][0]["message"]["content"]
    except Exception:
        return ""

def is_binary(data: bytes) -> bool:
    # Heuristic: if it has NUL bytes, treat as binary
    return b"\x00" in data

def deep_merge_json(base: Any, new: Any) -> Any:
    if isinstance(base, dict) and isinstance(new, dict):
        out = dict(base)
        for k, v in new.items():
            if k in out:
                out[k] = deep_merge_json(out[k], v)
            else:
                out[k] = v
        return out
    if isinstance(base, list) and isinstance(new, list):
        # dedupe by JSON string repr
        seen = set()
        out = []
        for item in base + new:
            key = json.dumps(item, sort_keys=True)
            if key not in seen:
                seen.add(key)
                out.append(item)
        return out
    # fallback: prefer new but keep both by embedding versions when types differ
    if type(base) is not type(new):
        return {"_versions": [base, new]}
    return new

def make_manifest(root: Path) -> Dict[str, Any]:
    return {
        "root": str(root),
        "ts_start": time.time(),
        "ops": [],
        "training_pairs": [],
    }

def add_training_pair(manifest: Dict[str, Any], file: str, prompt: str, completion: str):
    manifest["training_pairs"].append({
        "file": file,
        "prompt": prompt,
        "completion": completion,
    })

def hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def should_include(path: Path, include_globs: List[str], excludes: List[str]) -> bool:
    parts = set(path.parts)
    if any(e in parts for e in excludes):
        return False
    rel = str(path)
    return any(fnmatch.fnmatch(rel, g) for g in include_globs)

def process_text_file(path: Path, model: str, root: Path, manifest: Dict[str, Any], apply: bool, dry_run: bool, patches_dir: Path) -> Tuple[bool, str]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    user = f"File path: {path.as_posix()}\n\n{raw}"
    out = chat_complete(model, [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ])
    if not out:
        return False, "no_change"

    # training pair
    add_training_pair(manifest, str(path), SYSTEM_PROMPT, out)

    if out.strip() == raw.strip():
        return False, "no_change"

    # write patch suggestion
    rel = path.relative_to(root)
    patch_file = patches_dir / (rel.as_posix().replace("/", "_") + ".suggested.txt")
    patch_file.parent.mkdir(parents=True, exist_ok=True)
    patch_file.write_text(out, encoding="utf-8")

    if not apply or dry_run:
        manifest["ops"].append({"file": str(path), "action": "suggested_text", "patch": str(patch_file)})
        return True, "suggested"

    # attempt automatic merge as a simple replace with backup; also write a unified diff
    before = raw.splitlines(keepends=False)
    after = out.splitlines(keepends=False)
    diff = difflib.unified_diff(before, after, fromfile=str(path), tofile=str(path), lineterm="")
    diff_text = "\n".join(diff)

    backup = path.with_suffix(path.suffix + ".bak")
    backup.write_text(raw, encoding="utf-8")
    path.write_text(out, encoding="utf-8")

    diff_file = patches_dir / (rel.as_posix().replace("/", "_") + ".diff.txt")
    diff_file.write_text(diff_text, encoding="utf-8")

    manifest["ops"].append({
        "file": str(path),
        "action": "applied_text",
        "backup": str(backup),
        "patch": str(patch_file),
        "diff": str(diff_file),
    })
    return True, "applied"

def process_json_like(path: Path, model: str, root: Path, manifest: Dict[str, Any], apply: bool, dry_run: bool, patches_dir: Path) -> Tuple[bool, str]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    user = f"File path: {path.as_posix()}\n\n{raw}"
    out = chat_complete(model, [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ])
    if not out:
        return False, "no_change"

    add_training_pair(manifest, str(path), SYSTEM_PROMPT, out)

    # try to parse both versions and deep-merge; if either fails, fallback to suggest-only
    try:
        base = json.loads(raw)
        new = json.loads(out)
    except Exception:
        # just keep suggestion
        rel = path.relative_to(root)
        patch_file = patches_dir / (rel.as_posix().replace("/", "_") + ".suggested.txt")
        patch_file.parent.mkdir(parents=True, exist_ok=True)
        patch_file.write_text(out, encoding="utf-8")
        manifest["ops"].append({"file": str(path), "action": "suggested_json", "patch": str(patch_file)})
        return True, "suggested"

    merged = deep_merge_json(base, new)
    if merged == base:
        return False, "no_change"

    rel = path.relative_to(root)
    patch_file = patches_dir / (rel.as_posix().replace("/", "_") + ".merged.json")
    patch_file.parent.mkdir(parents=True, exist_ok=True)
    patch_file.write_text(json.dumps(merged, indent=2), encoding="utf-8")

    if not apply or dry_run:
        manifest["ops"].append({"file": str(path), "action": "suggested_merge", "patch": str(patch_file)})
        return True, "suggested"

    backup = path.with_suffix(path.suffix + ".bak")
    backup.write_text(raw, encoding="utf-8")
    path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    manifest["ops"].append({
        "file": str(path),
        "action": "applied_merge",
        "backup": str(backup),
        "patch": str(patch_file),
    })
    return True, "applied"

def run(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    if not root.exists():
        print(f"ERROR: root not found: {root}", file=sys.stderr)
        return 2

    forge_dir = root / ".forge"
    patches_dir = forge_dir / "patches"
    snaps_dir = forge_dir / "snapshots"
    data_dir = forge_dir / "data"
    for d in (patches_dir, snaps_dir, data_dir):
        d.mkdir(parents=True, exist_ok=True)

    if args.list_models:
        models = list_models()
        print(json.dumps({"models": models}, indent=2))
        return 0

    model = args.model
    if not model:
        models = list_models()
        if not models:
            print("ERROR: No models available from LM Studio. Start a model first.", file=sys.stderr)
            return 2
        # Choose a decent default if present
        preferred = [m for m in models if any(x in m.lower() for x in ("deepseek", "codestral", "mistral", "qwen"))]
        model = preferred[0] if preferred else models[0]
        print(f"Using model: {model}")

    manifest = make_manifest(root)
    file_count = 0
    changed = 0

    # snapshot of file list for reproducibility
    files: List[Path] = []
    for pat in args.include or DEFAULT_INCLUDE_GLOBS:
        for p in root.glob(pat):
            if p.is_file() and should_include(p.relative_to(root), args.include or DEFAULT_INCLUDE_GLOBS, args.exclude or DEFAULT_EXCLUDES):
                files.append(p)

    # Serialize list
    (forge_dir / "file_list.json").write_text(json.dumps([str(f.relative_to(root)) for f in files], indent=2), encoding="utf-8")

    for path in files:
        rel = path.relative_to(root)
        try:
            data = path.read_bytes()
        except Exception:
            continue

        if len(data) > args.max_bytes:
            continue
        if is_binary(data):
            continue

        file_count += 1
        suffix = path.suffix.lower()
        try:
            if suffix in (".json",):
                did, status = process_json_like(path, model, root, manifest, apply=not args.dry_run and args.apply, dry_run=args.dry_run, patches_dir=patches_dir)
            else:
                did, status = process_text_file(path, model, root, manifest, apply=not args.dry_run and args.apply, dry_run=args.dry_run, patches_dir=patches_dir)
            if did:
                changed += 1
        except Exception as e:
            manifest["ops"].append({"file": str(rel), "action": "error", "error": str(e)})

    manifest["ts_end"] = time.time()
    manifest["summary"] = {"files_seen": file_count, "changed": changed, "model": model, "root": str(root)}

    (forge_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"FORGE COMPLETE files={file_count} changed={changed} model={model} root={root}")
    return 0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Forge — LM Studio teacher orchestrator (merge-not-overwrite)")
    ap.add_argument("--root", default="/src", help="Repo root (default: /src)")
    ap.add_argument("--model", default=None, help="Model id to use (from LM Studio /v1/models)")
    ap.add_argument("--list-models", action="store_true", help="List models from LM Studio and exit")
    ap.add_argument("--apply", action="store_true", help="Apply changes (else write suggestions only)")
    ap.add_argument("--dry-run", action="store_true", help="Do not write changes; preview only")
    ap.add_argument("--max-bytes", type=int, default=800_000, help="Skip files larger than this")
    ap.add_argument("--include", nargs="*", default=None, help="Glob patterns to include (default: common source configs)")
    ap.add_argument("--exclude", nargs="*", default=None, help="Names to exclude (directories or path parts)")
    return ap.parse_args(argv)


if __name__ == "__main__":
    sys.exit(run(parse_args()))
