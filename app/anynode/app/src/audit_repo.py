
"""
audit_repo.py
Builds a lightweight import graph for a Python repo and flags likely-orphan modules.
Usage:
    python audit_repo.py "C:\Projects\LillithNew\src"
Outputs:
    import_graph.json, orphan_candidates.txt in a ./_reports folder under the root.
"""
import ast, os, sys, json, pathlib
from collections import defaultdict

def iter_py_files(root):
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.endswith(".py"):
                yield pathlib.Path(dirpath) / f

def module_name_from_path(root, path):
    rel = pathlib.Path(path).relative_to(root)
    parts = list(rel.parts)
    parts[-1] = parts[-1].replace(".py","")
    parts = [p for p in parts if p != "__pycache__" and p != "__init__"]
    return ".".join(parts)

def parse_imports(py_path):
    try:
        with open(py_path, "r", encoding="utf-8", errors="ignore") as fh:
            src = fh.read()
        tree = ast.parse(src)
    except Exception:
        return set()
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                out.add(n.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                out.add(node.module.split(".")[0])
    return out

def main(root):
    root = pathlib.Path(root)
    rpt = root / "_reports"
    rpt.mkdir(parents=True, exist_ok=True)

    files = list(iter_py_files(root))
    local_modules = set()
    for f in files:
        mn = module_name_from_path(root, f)
        if mn:
            local_modules.add(mn.split(".")[0])

    graph = defaultdict(list)
    reverse = defaultdict(set)

    for f in files:
        imports = parse_imports(f)
        me = module_name_from_path(root, f).split(".")[0]
        for imp in imports:
            if imp in local_modules and imp != me:
                graph[me].append(imp)
                reverse[imp].add(me)

    entrypoints = {
        "consciousness_orchestrator", "master_orchestrator",
        "nexus_bootstrap", "nexus_controller", "modal_app",
        "CogniKube_wrapper"
    }

    orphan = []
    for mod in sorted(local_modules):
        if mod in entrypoints:
            continue
        if len(reverse.get(mod, set())) == 0:
            orphan.append(mod)

    with open(rpt / "import_graph.json", "w", encoding="utf-8") as f:
        json.dump({"graph": graph, "reverse": {k:list(v) for k,v in reverse.items()}}, f, indent=2)

    with open(rpt / "orphan_candidates.txt", "w", encoding="utf-8") as f:
        for m in orphan:
            f.write(m + "\n")

    print(f"Analyzed {len(files)} files. Local modules: {len(local_modules)}")
    print(f"Orphan candidates: {len(orphan)} (see {rpt/'orphan_candidates.txt'})")

if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    main(root)
