#!/usr/bin/env python
import argparse, json, os, re, sys
from pathlib import Path

# ---------- helpers ----------
def load_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""

def camel_case(name: str) -> bool:
    return bool(re.match(r"[A-Z]", name))

def module_rel_from_dotted(dotted: str) -> str:
    return dotted.replace(".", os.sep) + ".py"

def rel_from_abs(abs_path: str, src_root: Path, dotted_fallback: str | None) -> str:
    try:
        a = Path(abs_path)
        return str(a.relative_to(src_root))
    except Exception:
        if dotted_fallback:
            return module_rel_from_dotted(dotted_fallback)
        return os.path.basename(abs_path)

def unique_actions(actions):
    seen = set()
    out = []
    for a in actions:
        key = (a["kind"], a.get("relPath",""), a.get("content",""), a.get("pattern",""), a.get("replacement",""))
        if key not in seen:
            seen.add(key)
            out.append(a)
    return out

# ---------- rule engine ----------
def build_actions_from_log(log_text: str, src_root: Path, rules: list[dict]) -> list[dict]:
    actions: list[dict] = []

    # Pass 1: regex DB
    for rule in rules:
        pat = re.compile(rule["pattern"], re.S)
        for m in pat.finditer(log_text):
            rtype = rule["type"]
            if rtype == "ModuleNotFound":
                dotted = m.group("dotted")
                rel = module_rel_from_dotted(dotted)
                actions.append({"kind":"touch","relPath":rel})
                # make minimal stub file if empty
                content = f'"""auto-stub for {dotted}"""\n'
                actions.append({"kind":"append","relPath":rel,"content":content})
            elif rtype == "CannotImportName":
                sym  = m.group("sym")
                pkg  = m.group("pkg")
                path = m.group("path")
                rel  = rel_from_abs(path, src_root, pkg)
                if camel_case(sym):
                    stub = f"class {sym}:\n    def __init__(self,*a,**k): pass\n    def start(self,*a,**k): return True\n"
                else:
                    stub = f"def {sym}(*a,**k):\n    return None\n"
                actions.append({"kind":"append","relPath":rel,"content":stub})
            elif rtype == "ImportErrorName":
                # variant without a file path
                sym  = m.group("sym")
                pkg  = m.group("pkg")
                rel  = module_rel_from_dotted(pkg)
                if camel_case(sym):
                    stub = f"class {sym}:\n    def __init__(self,*a,**k): pass\n    def start(self,*a,**k): return True\n"
                else:
                    stub = f"def {sym}(*a,**k):\n    return None\n"
                actions.append({"kind":"append","relPath":rel,"content":stub})
            elif rtype == "SyntaxUnmatchedBrace":
                # crude heal: comment offending line (if we can grab filename/line)
                # This generic rule doesn’t include filename—leave to LLM stage.
                pass

    # Pass 2: hard-coded quality-of-life shims (safe & idempotent)
    # Manifest server shim
    rel = module_rel_from_dotted("Systems.engine.api.manifest_server")
    shim = (
        '"""shim for start_manifest_server"""\n'
        "import importlib\n"
        "__deep_chosen__ = None\n"
        "for _name, _kind in [\n"
        "  ('service.cogniKubes.anynode.files.manifest_server','ASGI'),\n"
        "  ('service.cogniKubes.heart.files.manifest_server','func'),\n"
        "  ('service.cogniKubes.edge_anynode.files.manifest_server','func'),\n"
        "]:\n"
        "    try:\n"
        "        _m = importlib.import_module(_name)\n"
        "    except Exception:\n"
        "        continue\n"
        "    if hasattr(_m,'start_manifest_server'):\n"
        "        start_manifest_server = getattr(_m,'start_manifest_server')\n"
        "        __deep_chosen__ = _name + (' (ASGI)' if _kind=='ASGI' else '')\n"
        "        break\n"
        "    for alt in ('start','main','run','serve','start_server','start_api','launch'):\n"
        "        if hasattr(_m,alt) and callable(getattr(_m,alt)):\n"
        "            def start_manifest_server(*a,**k):\n"
        "                return getattr(_m,alt)(*a,**k)\n"
        "            __deep_chosen__ = _name + '.app (ASGI)' if 'app' in dir(_m) else _name\n"
        "            break\n"
    )
    actions.append({"kind":"write","relPath":rel,"content":shim})

    # Pulse core shim
    rel = module_rel_from_dotted("Systems.engine.pulse.pulse_core")
    shim2 = (
        '"""shim for initialize_pulse_core"""\n'
        "import importlib\n"
        "__deep_chosen__ = None\n"
        "for _name, _sym in [\n"
        "  ('service.cogniKubes.heart.files.pulse_core','PulseCore'),\n"
        "  ('service.cogniKubes.anynode.files.pulse_core','PulseCore'),\n"
        "]:\n"
        "    try:\n"
        "        _m = importlib.import_module(_name)\n"
        "    except Exception:\n"
        "        continue\n"
        "    if hasattr(_m,_sym):\n"
        "        def initialize_pulse_core(*a,**k):\n"
        "            return getattr(_m,_sym)(*a,**k)\n"
        "        __deep_chosen__ = _name + '.' + _sym\n"
        "        break\n"
    )
    actions.append({"kind":"write","relPath":rel,"content":shim2})

    # Pulse timer/listener shims
    for mod, cls in [
        ("Systems.engine.pulse.pulse_timer", "PulseTimer"),
        ("Systems.engine.pulse.pulse_listener", "PulseListener"),
    ]:
        rel = module_rel_from_dotted(mod)
        body = (
            f'"""shim for {cls}"""\n'
            "import importlib\n"
            "for _name in [\n"
            "  'service.cogniKubes.heart.files.pulse_timer',\n"
            "  'service.cogniKubes.anynode.files.pulse_timer',\n"
            "  'service.cogniKubes.heart.files.pulse_listener',\n"
            "  'service.cogniKubes.anynode.files.pulse_listener',\n"
            "]:\n"
            "    try:\n"
            "        _m = importlib.import_module(_name)\n"
            "    except Exception:\n"
            "        continue\n"
            f"    if hasattr(_m,'{cls}'):\n"
            f"        {cls} = getattr(_m,'{cls}')\n"
            "        break\n"
        )
        actions.append({"kind":"write","relPath":rel,"content":body})

    return unique_actions(actions)

def maybe_llm_assist(log_text: str, src_root: Path, llm_url: str|None, llm_model: str|None) -> list[dict]:
    """
    Optional: query local LM Studio to suggest targeted diffs.
    Returns list of actions (append/write/replace_regex).
    """
    if not llm_url or not llm_model:
        return []

    import urllib.request
    payload = {
        "model": llm_model,
        "messages": [
            {"role":"system","content":"You are a senior Python build doctor. Output JSON ONLY with an 'actions' array. Each action has: kind (append|write|replace_regex), relPath (relative to src root), and one of: content OR (pattern,replacement). No prose."},
            {"role":"user","content": f"Project src root: {src_root}\nBoot error log:\n{log_text}\nInfer the minimal corrective edits to make imports resolvable. If a module is missing a symbol, append a minimal stub. If a shim is needed, write it. Use replace_regex only if a single-line syntax is clearly bad."}
        ],
        "temperature": 0.2,
        "max_tokens": 500
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(llm_url, data=data, headers={"Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            raw = r.read().decode("utf-8","ignore")
        resp = json.loads(raw)
        msg = resp.get("choices",[{}])[0].get("message",{}).get("content","")
        msg = msg.strip()
        # Try to extract JSON if model wrapped it in fencing
        m = re.search(r"\{[\s\S]*\}\s*$", msg)
        if m:
            obj = json.loads(m.group(0))
            acts = obj.get("actions", [])
            # basic sanity filter
            out = []
            for a in acts:
                if a.get("kind") in ("append","write","replace_regex") and a.get("relPath"):
                    out.append(a)
            return out
    except Exception:
        return []
    return []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--rules", required=True)
    ap.add_argument("--llm-url", default=None)
    ap.add_argument("--llm-model", default=None)
    args = ap.parse_args()

    src_root = Path(args.src)
    log_text = load_text(Path(args.log))
    rules = json.loads(load_text(Path(args.rules))).get("rules", [])

    actions = build_actions_from_log(log_text, src_root, rules)
    # Optional: LLM proposals appended at the end (deduped later by PS side)
    actions.extend(maybe_llm_assist(log_text, src_root, args.llm_url, args.llm_model))

    print(json.dumps({"actions": actions}, ensure_ascii=False))

if __name__ == "__main__":
    main()
