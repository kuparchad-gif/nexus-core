# ci/check_subjects.py
"""
Nexus CI Policy Check

- Validates subject strings across the repo follow the canonical schema:
  <plane>.<mesh>.<kind>.<domain>.<...>
  plane := svc|sys|cog|play
  mesh  := sense|think|heal|archive
  kind  := cap.register|cap.request|cap.reply|events|metrics

- Validates edge-anynode.policy.json if present.

- Guards /etc vs /var conventions in code/config:
  - Fail if the repo writes runtime files to /etc or loads configs from /var.

Outputs a short report to ci/_report.txt and exits nonzero on violations.
"""

import os, re, json, sys
from pathlib import Path

ROOT = Path(".")
REPORT = Path("ci/_report.txt")
REPORT.parent.mkdir(parents=True, exist_ok=True)

SUBJECT_RE = re.compile(
    r'(?P<subj>\b(?:svc|sys|cog|play)\.(?:sense|think|heal|archive)\.(?:cap\.register|cap\.request|cap\.reply|events|metrics)\.[A-Za-z0-9_\-\.>]+)'
)

BAD_ETC_WRITE_RE = re.compile(r'open\(.*/etc/.*["\']\s*,\s*["\'](?:w|a)\b', re.IGNORECASE)
BAD_VAR_CONFIG_RE = re.compile(r'/(?:var|VAR)[/\\](?:etc|config)', re.IGNORECASE)

errors = []
warnings = []

def scan_file(path: Path):
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return

    # Find subject-like tokens and validate by regex (already ensures shape)
    for m in SUBJECT_RE.finditer(text):
        subj = m.group("subj")
        # Additional guard: no double dots, no accidental spaces
        if ".." in subj or " " in subj:
            errors.append(f"{path}: invalid subject formatting: {subj}")

    # Prohibit writes to /etc (very naive heuristic)
    if BAD_ETC_WRITE_RE.search(text):
        errors.append(f"{path}: code appears to WRITE to /etc (forbidden for runtime data)")

    # Prohibit loading config from /var paths
    if re.search(r'/(?:var|VAR)/.*(?:\.env|config|\.ini|\.yaml|\.yml)$', text):
        errors.append(f"{path}: code appears to load CONFIG from /var (configs belong in /etc)")

    # Warn if subjects without plane prefixes are found (legacy 'mesh.*' etc.)
    if re.search(r'(^|\W)(?:mesh\.(?:sense|think|heal|archive)|cap\.request\.)', text):
        warnings.append(f"{path}: possible legacy subject usage detected (missing plane prefix?)")

def validate_edge_policy():
    p = ROOT / "policy" / "edge-anynode.policy.json"
    if not p.exists():
        warnings.append("edge-anynode.policy.json not found under policy/.")
        return
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        errors.append(f"{p}: invalid JSON: {e}")
        return

    default = obj.get("default")
    if default not in ("deny","allow"):
        errors.append(f"{p}: default must be 'deny' or 'allow'")
    for i, rule in enumerate(obj.get("allow", []), start=1):
        frm = rule.get("from")
        to  = rule.get("to")
        def _ok(subj):
            if subj is None: 
                return True
            # allow '>' wildcards; temporarily replace to validate shape
            subj = str(subj).replace(">", "events.x")
            return bool(SUBJECT_RE.match(subj))
        if isinstance(frm, list):
            for s in frm:
                if not _ok(s):
                    errors.append(f"{p} allow[{i}].from invalid: {s}")
        else:
            if not _ok(frm):
                errors.append(f"{p} allow[{i}].from invalid: {frm}")
        if not _ok(to):
            errors.append(f"{p} allow[{i}].to invalid: {to}")

def main():
    # Gather files to scan (skip common vendor/build dirs)
    exclude = {"node_modules", ".git", ".venv", "dist", "build", "__pycache__", ".mypy_cache", ".pytest_cache", ".idea", ".vscode"}
    exts = {".py", ".ts", ".js", ".json", ".yaml", ".yml", ".toml", ".ini", ".md", ".ps1", ".sh", ".env", ".txt"}

    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        parts = set(path.parts)
        if parts & exclude:
            continue
        if path.suffix.lower() not in exts:
            continue
        scan_file(path)

    validate_edge_policy()

    with REPORT.open("w", encoding="utf-8") as f:
        if errors:
            f.write("Nexus Policy Check: FAIL\n\n")
            for e in errors:
                f.write("ERROR: " + e + "\n")
        else:
            f.write("Nexus Policy Check: PASS\n")
        if warnings:
            f.write("\nWarnings:\n")
            for w in warnings:
                f.write("WARN: " + w + "\n")

    print(REPORT.read_text(encoding="utf-8"))
    sys.exit(1 if errors else 0)

if __name__ == "__main__":
    main()
