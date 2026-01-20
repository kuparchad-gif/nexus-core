# ci/validate_categories.py
"""
Validates that:
  1) Every cap.register payload includes a valid 'category' (one of six).
  2) Optional: mapping.services.json entries also conform.
  3) Subjects (if present) recommend a category-prefixed domain (cons|subc|upcg|mem|core|edge).

Usage:
  python ci/validate_categories.py [--strict]
  - If --strict is given, fail when any subject domain is missing the category prefix.
"""

import os, sys, json, re
from pathlib import Path

CATS = ["Consciousness","Subconsciousness","Upper Cognition","Memory","Nexus Core","Edge Anynode"]
CAT_KEYS = {"Consciousness":"cons","Subconsciousness":"subc","Upper Cognition":"upcg","Memory":"mem","Nexus Core":"core","Edge Anynode":"edge"}

SUBJECT_RE = re.compile(r'^(svc|sys|cog|play)\.(sense|think|heal|archive)\.(cap\.register|cap\.request|cap\.reply|events|metrics)\.([A-Za-z0-9_.>\-]+)$')

STRICT = "--strict" in sys.argv

errors = []
warnings = []

def check_payload(obj, path):
    if not isinstance(obj, dict): return
    has_keys = {"service","caps","plane","mesh"} <= obj.keys()
    if has_keys:
        cat = obj.get("category")
        if cat not in CATS:
            errors.append(f"{path}: missing/invalid category: {cat}")

def scan_repo():
    # 1) cap.register JSONs
    for p in Path(".").rglob("*.json"):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        # Heuristic: treat as cap.register if it has mandatory keys
        if isinstance(obj, dict) and {"service","caps","plane","mesh"} <= obj.keys():
            check_payload(obj, str(p))

    # 2) mapping.services.json (if present)
    mp = Path("mapping.services.json")
    if mp.exists():
        try:
            m = json.loads(mp.read_text(encoding="utf-8"))
            for i, s in enumerate(m.get("services", []), 1):
                if s.get("category") not in CATS:
                    errors.append(f"mapping.services.json[{i}]: invalid category: {s.get('category')}")
        except Exception as e:
            errors.append(f"mapping.services.json: unreadable: {e}")

    # 3) subject domain prefixes (recommendation unless --strict)
    for p in Path(".").rglob("*"):
        if p.suffix.lower() not in {".py",".ts",".js",".md",".ps1",".sh",".yaml",".yml"}:
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for m in SUBJECT_RE.finditer(txt):
            domain = m.group(4)
            if not domain.startswith(("cons.","subc.","upcg.","mem.","core.","edge.")):
                msg = f"{p}: subject domain not prefixed with category: {m.group(0)}"
                (errors if STRICT else warnings).append(msg)

scan_repo()

report = Path("ci/_category_report.txt")
report.parent.mkdir(parents=True, exist_ok=True)
with report.open("w", encoding="utf-8") as f:
    if errors:
        f.write("Category validation: FAIL\n")
        for e in errors: f.write("ERROR: " + e + "\n")
    else:
        f.write("Category validation: PASS\n")
    if warnings:
        f.write("\nWarnings:\n")
        for w in warnings: f.write("WARN: " + w + "\n")

print(report.read_text(encoding="utf-8"))
sys.exit(1 if errors else 0)
