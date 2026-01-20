from pathlib import Path
import json

def scan_modules(base_path="C:/Engineers/root/app"):
    modules = {}
    for py_file in Path(base_path).rglob("*.py"):
        try:
            with open(py_file, encoding="utf-8") as f:
                content = f.read(1000)
            modules[str(py_file)] = {"preview": content[:500]}
        except Exception:
            continue
    with open("module_index.json", "w", encoding="utf-8") as f:
        json.dump(modules, f, indent=2)
    return modules
