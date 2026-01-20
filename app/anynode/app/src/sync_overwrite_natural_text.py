#!/usr/bin/env python3
"""
sync_overwrite_natural_text.py

Scan a source directory and a destination directory.
- If a file from source doesn't exist in destination -> copy it.
- If a file exists in destination -> overwrite ONLY if the destination looks like "regular English text" (not code).
- Never moves files; always writes to destination path.
- Atomic writes; optional dry-run.

Usage:
  python sync_overwrite_natural_text.py --src /path/to/src --dst /path/to/dst --dry-run
"""

import argparse
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Tuple

# Common code/config/extensions we should treat as "code-like"
CODE_EXTS = {
    ".py", ".pyw", ".ipynb", ".r", ".jl",
    ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx",
    ".java", ".kt", ".scala", ".go", ".rs", ".rb", ".php",
    ".c", ".h", ".hpp", ".hh", ".cc", ".cpp", ".ino",
    ".cs", ".fs",
    ".sh", ".bash", ".zsh", ".ps1", ".bat", ".cmd",
    ".sql", ".pl", ".pm", ".lua",
    ".html", ".xhtml", ".htm", ".xml",
    ".css", ".scss", ".less",
    ".json", ".jsonl", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".env", ".editorconfig",
    ".gradle", ".mdx",
    ".proto", ".thrift", ".avsc",
    ".tex", ".bib",
}

# Usually human prose; still analyzed by content to be safe
LIKELY_TEXT_EXTS = {".txt", ".md", ".rst", ".rtf"}

# Heuristic regexes for code-y lines
CODEY_PATTERNS = re.compile(
    r"""
    (\bdef\b|\bclass\b|\bimport\b|\bfrom\b|\breturn\b|\bfunc(?:tion)?\b|\bvar\b|\blet\b|\bconst\b|
     \bpublic\b|\bprivate\b|\bstatic\b|\bstruct\b|\btemplate\b|
     #includes?|->|=>|::|::=|==|!=|<=|>=|
     [{}`();<>]|</?[a-zA-Z][a-zA-Z0-9-]*\s*[^>]*>|
     \bSELECT\b|\bINSERT\b|\bUPDATE\b|\bDELETE\b|\bCREATE\b|\bTABLE\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")

def is_binary_bytes(b: bytes) -> bool:
    """Simple binary sniff: NULL bytes, very high non-text ratio."""
    if b is None or len(b) == 0:
        return False
    if b.find(b"\x00") != -1:
        return True
    # if >30% bytes are non-printable (excluding common whitespace), call it binary
    text_chars = bytearray({7,8,9,10,12,13,27} | set(range(0x20, 0x100)))
    nontext = sum(c not in text_chars for c in b)
    return (nontext / max(1, len(b))) > 0.30

def read_sample(path: Path, max_bytes: int = 512 * 1024) -> Tuple[str, bytes]:
    """Read up to max_bytes safely; return (text, raw)."""
    raw = b""
    try:
        with open(path, "rb") as f:
            raw = f.read(max_bytes)
    except Exception:
        return ("", b"")
    try:
        text = raw.decode("utf-8", errors="ignore")
    except Exception:
        text = ""
    return text, raw

def english_text_score(text: str) -> float:
    """
    Crude 'English-likeness' score: ratio of alphabetic words to tokens,
    penalized by code-like density.
    """
    if not text:
        return 0.0
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0.0

    tokens = re.findall(r"[A-Za-z']+", text)
    words = [t for t in tokens if len(t) >= 2]
    word_ratio = (len(words) / max(1, len(tokens))) if tokens else 0.0

    codey_lines = sum(1 for ln in lines if CODEY_PATTERNS.search(ln))
    code_ratio = codey_lines / max(1, len(lines))

    html_ratio = sum(1 for ln in lines if HTML_TAG.search(ln)) / max(1, len(lines))

    # Score: prefer high word_ratio, low code/html ratios
    score = word_ratio - 0.6 * code_ratio - 0.3 * html_ratio
    return max(0.0, min(1.0, score))

def classify_destination(path: Path) -> str:
    """
    Classify destination file as 'code', 'text', 'binary', or 'unknown'.
    - Extension-first for known types.
    - Otherwise content heuristics.
    """
    ext = path.suffix.lower()

    # Fast path by extension
    if ext in CODE_EXTS:
        return "code"
    if ext in LIKELY_TEXT_EXTS:
        # Still sanity-check content
        txt, raw = read_sample(path)
        if is_binary_bytes(raw):
            return "binary"
        score = english_text_score(txt)
        return "text" if score >= 0.5 else "code"

    # Fallback by content
    txt, raw = read_sample(path)
    if is_binary_bytes(raw):
        return "binary"

    # Code-ish?
    lines = [ln for ln in txt.splitlines() if ln.strip()]
    codey = sum(1 for ln in lines if CODEY_PATTERNS.search(ln))
    code_ratio = codey / max(1, len(lines))
    if code_ratio >= 0.20:
        return "code"

    # English-ish?
    if english_text_score(txt) >= 0.55:
        return "text"

    return "unknown"

def ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def atomic_write(dst: Path, data: bytes):
    """Write atomically: temp file then replace."""
    ensure_parent(dst)
    with tempfile.NamedTemporaryFile(delete=False, dir=str(dst.parent)) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    tmp_path.replace(dst)

def process_file(src_file: Path, dst_root: Path, dry_run: bool = False, verbose: bool = True):
    rel = src_file.relative_to(src_root)
    dst_file = dst_root / rel

    # Skip directories explicitly
    if src_file.is_dir():
        return

    # Read source content (pass-through if binary)
    src_bytes = b""
    with open(src_file, "rb") as f:
        src_bytes = f.read()

    if not dst_file.exists():
        if verbose:
            print(f"[COPY] {rel}  (new)")
        if not dry_run:
            ensure_parent(dst_file)
            shutil.copy2(src_file, dst_file)
        return

    # Destination exists -> only overwrite if destination is plain English text
    cls = classify_destination(dst_file)
    if cls == "text":
        if verbose:
            print(f"[OVERWRITE] {rel}  (dest is text)")
        if not dry_run:
            atomic_write(dst_file, src_bytes)
    else:
        if verbose:
            print(f"[SKIP] {rel}  (dest classified as {cls})")

def scan_and_sync(src_root: Path, dst_root: Path, dry_run: bool = False, verbose: bool = True):
    if not src_root.exists() or not src_root.is_dir():
        print(f"ERROR: src '{src_root}' is not a directory", file=sys.stderr)
        sys.exit(2)
    if not dst_root.exists():
        if verbose:
            print(f"[MKDIR] {dst_root}")
        dst_root.mkdir(parents=True, exist_ok=True)

    for path in src_root.rglob("*"):
        if path.is_file():
            process_file(path, dst_root, dry_run=dry_run, verbose=verbose)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Copy from src to dst; overwrite only when destination is plain English text (not code).")
    ap.add_argument("--src", required=True, type=Path, help="Source directory")
    ap.add_argument("--dst", required=True, type=Path, help="Destination directory")
    ap.add_argument("--dry-run", action="store_true", help="Show actions without writing")
    ap.add_argument("--quiet", action="store_true", help="Suppress logs")
    args = ap.parse_args()

    src_root = args.src.resolve()
    dst_root = args.dst.resolve()
    scan_and_sync(src_root, dst_root, dry_run=args.dry_run, verbose=not args.quiet)

