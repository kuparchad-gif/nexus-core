import sys, io, os, time, difflib, shutil
from pathlib import Path

def apply_unified_diff(diff_text: str, root: Path) -> list[str]:
    """
    Apply unified diff(s) produced against files under `root`.
    Returns list of touched files.
    """
    touched = []
    chunks = diff_text.split("\ndiff --git ")
    for ch in chunks:
        if not ch.strip(): 
            continue
        # try to find '+++ b/<path>' line
        tgt = None
        for line in ch.splitlines():
            if line.startswith("+++ ") and ("\\dev\\null" not in line):
                # formats: +++ b/path OR +++ path
                tgt = line.split(" ",1)[1].strip()
                if tgt.startswith("b/"): tgt = tgt[2:]
                break
        if not tgt: 
            # fallback: look for ---/+++ pair
            continue
        file_path = (root / tgt).resolve()
        if root not in file_path.parents and file_path != root:
            raise RuntimeError(f"Refusing to patch outside root: {file_path}")
        # reconstruct old/new from diff
        # simpler approach: read file, build patched with difflib.restore
        old = file_path.read_text(encoding="utf-8") if file_path.exists() else ""
        # extract only hunk lines beginning with ' ', '+', '-'
        hunk = []
        capture = False
        for line in ch.splitlines():
            if line.startswith('@@ '):
                capture = True
                continue
            if capture:
                if line and line[0] in " +-":
                    hunk.append(line)
        # difflib.restore expects lines with ' ' or '-'
        # Instead, use patch-like rebuild:
        new_lines = []
        old_lines = old.splitlines(keepends=False)
        i = 0
        for ln in hunk:
            if ln.startswith(' '):
                new_lines.append(old_lines[i]); i += 1
            elif ln.startswith('-'):
                i += 1
            elif ln.startswith('+'):
                new_lines.append(ln[1:])
        # append remaining tails if any context didn’t cover all
        new_lines.extend(old_lines[i:])
        new_text = "\n".join(new_lines) + ("\n" if new_lines and not new_lines[-1].endswith("\n") else "")
        # backup then write
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if file_path.exists():
            bak = file_path.with_suffix(file_path.suffix + f".bak.{int(time.time())}")
            shutil.copy2(file_path, bak)
        file_path.write_text(new_text, encoding="utf-8")
        touched.append(str(file_path))
    return touched

if __name__ == "__main__":
    root = Path(sys.argv[1]).resolve()
    diff_text = sys.stdin.read()
    changed = apply_unified_diff(diff_text, root)
    print("\n".join(changed))
