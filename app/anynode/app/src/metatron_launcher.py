#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Metatron One‑Click Launcher (enhanced)
- Cross‑OS: Windows/macOS/Linux
- Ensures Podman + podman machine (on Win/macOS)
- Ensures Python deps required by the launcher itself (PyYAML)
- Locates repo root that contains 'infra/podman/metatron.pod.yaml'
- Plays Kube (Podman) OR compiles CogniKubes (*.ckube.yaml) into a Pod spec and plays it
- Idempotent: safe to run multiple times
"""
import os, sys, platform, subprocess, shutil, time, pathlib, tempfile, argparse

try:
    import yaml
except Exception:
    yaml = None

ROOT = pathlib.Path(__file__).resolve().parent

def run(cmd, check=True, shell=False, env=None):
    print("+", cmd if isinstance(cmd,str) else " ".join(cmd))
    return subprocess.run(cmd, check=check, shell=shell, env=env)

def have(cmd):
    return shutil.which(cmd) is not None

def os_name():
    s = platform.system().lower()
    if s.startswith("win"): return "windows"
    if s == "darwin": return "macos"
    if s == "linux": return "linux"
    return s

# ------------------ Python deps for the launcher itself ------------------
def ensure_python_packages(pkgs):
    python = sys.executable
    # On Windows, prefer -m pip to avoid PATH weirdness
    for p in pkgs:
        try:
            __import__(p if p != "pyyaml" else "yaml")
            continue
        except Exception:
            pass
        print(f"Installing missing Python package: {p}")
        # Try to use pip in the current interpreter
        try:
            run([python, "-m", "pip", "install", "--upgrade", "pip"], check=False)
            run([python, "-m", "pip", "install", "--no-cache-dir", p], check=True)
        except Exception as e:
            print(f"WARNING: failed to install {p}: {e}")

# ------------------ Podman install & machine prep ------------------
def ensure_podman():
    if have("podman"):
        return True
    sys_os = os_name()
    print("Podman not found. Attempting automatic install...")

    try:
        if sys_os == "windows":
            if have("winget"):
                # CLI then Desktop for machine mgmt & Docker API shim
                run(["winget","install","-e","--id","RedHat.Podman",
                     "--accept-package-agreements","--accept-source-agreements","--silent"], check=False)
                run(["winget","install","-e","--id","RedHat.Podman-Desktop",
                     "--accept-package-agreements","--accept-source-agreements","--silent"], check=False)
            elif have("choco"):
                run(["choco","install","podman","-y"], check=False)
                run(["choco","install","podman-desktop","-y"], check=False)
            else:
                print("No winget/choco detected. Please install Podman Desktop from https://podman.io/")
        elif sys_os == "macos":
            if have("brew"):
                run(["brew","install","podman"], check=False)
                # Recommend Desktop for convenience (optional)
                run(["brew","install","--cask","podman-desktop"], check=False)
            else:
                print("Homebrew not found. Install Homebrew or Podman Desktop from https://podman.io/")
        elif sys_os == "linux":
            if have("apt"):
                run(["sudo","apt","update"], check=False)
                run(["sudo","apt","install","-y","podman"], check=False)
            elif have("dnf"):
                run(["sudo","dnf","install","-y","podman"], check=False)
            elif have("yum"):
                run(["sudo","yum","install","-y","podman"], check=False)
            elif have("pacman"):
                run(["sudo","pacman","-Sy","--noconfirm","podman"], check=False)
            else:
                print("Please install Podman via your distro package manager.")
    except Exception as e:
        print(f"Auto-install failed: {e}")
    return have("podman")

def ensure_machine():
    if os_name() == "linux":
        return
    # Make sure the Podman VM exists & is running
    run(["podman","machine","init","--cpus=4","--memory=4096","--disk-size=20"], check=False)
    run(["podman","machine","start"], check=False)
    # Rootful improves port binding parity
    try:
        run(["podman","machine","set","--rootful","true"], check=False)
    except Exception:
        pass

def ensure_network(name="metanet"):
    cp = subprocess.run(["podman","network","exists",name])
    if cp.returncode != 0:
        subprocess.run(["podman","network","create",name], check=False)

def podman_ok():
    try:
        run(["podman","info"], check=True)
        return True
    except Exception:
        return False

# ------------------ Repo discovery ------------------
def has_infra(p):
    return (p/"infra"/"podman"/"metatron.pod.yaml").exists()

def find_repo_root(start: pathlib.Path):
    # 1) current dir up
    cur = start
    for _ in range(5):
        if has_infra(cur): return cur
        cur = cur.parent
    # 2) common extracted folder names
    candidates = [
        start/"LillithWired",
        start/"lillith-wired",
        start/"nexus-metatron",
        start/"metatron",
        start/"Lillith"
    ]
    for c in candidates:
        if has_infra(c): return c
    # 3) deep scan one level
    for child in start.iterdir():
        if child.is_dir() and has_infra(child): return child
    return start

# ------------------ Building & playing ------------------
def build_images(repo_root: pathlib.Path):
    env = os.environ.copy()
    env["DOCKER_BUILDKIT"] = "1"
    pdir = repo_root/"infra"/"podman"
    # Build what exists; skip if Containerfile missing
    def maybe_build(cf, tag):
        f = pdir/cf
        if f.exists():
            run(["podman","build","-f", str(f), "-t", tag, str(repo_root)], check=True, env=env)
    maybe_build("Containerfile.core",           "localhost/metatron-core:local")
    maybe_build("Containerfile.edge",           "localhost/metatron-edge:local")
    maybe_build("Containerfile.viren",          "localhost/metatron-viren:local")
    maybe_build("Containerfile.lillith",        "localhost/metatron-lillith:local")
    maybe_build("Containerfile.memory",         "localhost/metatron-memory:local")
    maybe_build("Containerfile.consciousness",  "localhost/metatron-consciousness:local")
    maybe_build("Containerfile.subconsciousness","localhost/metatron-subcon:local")
    maybe_build("Containerfile.monitor",        "localhost/metatron-monitor:local")

def play_kube(repo_root: pathlib.Path, network="metanet"):
    k = repo_root/"infra"/"podman"/"metatron.pod.yaml"
    if not k.exists():
        print(f"ERROR: {k} not found.")
        sys.exit(10)
    run(["podman","play","kube","--network",network,str(k)], check=False)

# ------------------ CKube (optional) ------------------
def load_ckubes(dirpath):
    if yaml is None:
        print("Missing PyYAML; cannot load CKubes (*.ckube.yaml). Run: python -m pip install pyyaml")
        return []
    base = pathlib.Path(dirpath)
    files = sorted([p for p in base.glob("*.ckube.yaml")])
    specs = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            specs.append(yaml.safe_load(fh))
    return specs

def build_local_image(tag, svc_dir, entry):
    dockerfile = f"""FROM python:3.12-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir --upgrade pip && \
    (test -f requirements.txt && pip install --no-cache-dir -r requirements.txt || true) && \
    pip install --no-cache-dir fastapi uvicorn[standard] httpx pydantic
ENV PYTHONUNBUFFERED=1
CMD ["python","{entry}"]
"""
    with tempfile.TemporaryDirectory() as td:
        tdp = pathlib.Path(td)
        (tdp/"Dockerfile").write_text(dockerfile, encoding="utf-8")
        subprocess.run(["podman","build","-f", str(tdp/"Dockerfile"), "-t", tag, str(svc_dir)], check=True)

def compile_ckube_to_podman(spec, repo_root, rebuild=True, recreate=False, network="metanet"):
    if yaml is None:
        print("Cannot compile CKube without PyYAML.")
        return
    name = spec["name"]
    containers = []
    for x in (spec.get("experts") or []):
        img = None
        if "source" in x and str(x["source"]).startswith("path:"):
            svc_rel = str(x["source"]).split("path:",1)[1]
            svc_dir = (repo_root / svc_rel).resolve()
            img = f"localhost/{name}-{x['id']}:ckube"
            if rebuild:
                build_local_image(img, svc_dir, x.get("entry","app.py"))
        else:
            img = x.get("image")
        env = [{"name":k,"value":str(v)} for k,v in (x.get("env") or {}).items()]
        ports = [{"containerPort": int(x.get("port")), "hostPort": int(x.get("port")), "protocol": "TCP"}] if x.get("port") else []
        containers.append({"name": x["id"], "image": img, "env": env, "ports": ports})
    for s in (spec.get("sidecars") or []):
        ports = [{"containerPort": int(s.get("port")), "hostPort": int(s.get("port")), "protocol": "TCP"}] if s.get("port") else []
        containers.append({"name": s["id"], "image": s["image"], "args": s.get("args"), "ports": ports})

    pod = {
        "apiVersion":"v1","kind":"Pod",
        "metadata":{"name": name, "labels":{"ckube":"1","app":name}},
        "spec":{"containers": containers, "volumes": []}
    }
    with tempfile.TemporaryDirectory() as td:
        k = pathlib.Path(td)/f"{name}.pod.yaml"
        import yaml as _y
        k.write_text(_y.safe_dump(pod, sort_keys=False), encoding="utf-8")
        if recreate:
            subprocess.run(["podman","pod","rm","-f", name], check=False)
        subprocess.run(["podman","play","kube","--network", network, str(k)], check=False)

# ------------------ main ------------------
def main():
    ap = argparse.ArgumentParser(description="Metatron One‑Click Launcher")
    ap.add_argument("--repo-root", default=".", help="Path that contains 'infra/podman'")
    ap.add_argument("--ckube-dir", default=None, help="If provided, launch from *.ckube.yaml files in this dir")
    ap.add_argument("--skip-build", action="store_true", help="Skip building local images")
    ap.add_argument("--recreate", action="store_true", help="Recreate pod(s)")
    args = ap.parse_args()

    ensure_python_packages(["pyyaml"])  # launcher needs only PyYAML

    if not ensure_podman():
        print("Podman unavailable. Install it and re‑run.")
        sys.exit(2)
    ensure_machine()
    if not podman_ok():
        print("Podman not ready. Fix 'podman machine' and retry.")
        sys.exit(3)
    ensure_network("metanet")

    repo_root = pathlib.Path(args.repo_root).resolve()
    if not (repo_root/"infra"/"podman").exists():
        repo_root = find_repo_root(ROOT)

    if args.ckube_dir:
        specs = load_ckubes(args.ckube_dir) if yaml else []
        for spec in specs:
            compile_ckube_to_podman(spec, repo_root, rebuild=(not args.skip_build), recreate=args.recreate)
        print("CKubes launched.")
    else:
        if not args.skip_build:
            build_images(repo_root)
        if args.recreate:
            subprocess.run(["podman","pod","rm","-f","metatron"], check=False)
        play_kube(repo_root)

    print("\\nMetatron is launching. Endpoints (host):")
    print("  Core (gateway):   http://127.0.0.1:1313/health")
    print("  Qdrant:           http://127.0.0.1:6333/collections")
    print("  Loki:             http://127.0.0.1:3100/ready")
    print("  Consciousness:    http://127.0.0.1:8001/health")
    print("  Memory:           http://127.0.0.1:8003/health")
    print("\\nTip: 'podman pod ps' and 'podman ps --pod' to inspect. 'podman logs <ctr>' for logs.")

if __name__ == "__main__":
    main()
