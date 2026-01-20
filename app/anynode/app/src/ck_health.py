# scripts/ck_health.py
import os, json, argparse
from lilith.mesh.cognikube_registry import registry, client

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default=os.environ.get("COGNIKUBES_CFG","config/cognikubes.json"))
    args = ap.parse_args()

    os.environ["COGNIKUBES_CFG"] = args.cfg
    r = registry()
    print("Kubes:")
    for row in r.list():
        print(json.dumps(row, indent=2))

    # sample route
    c = client()
    res = c.route(role="spine", path="/health", payload={}, capability="relay")
    print("\nRoute test:", res)

if __name__ == "__main__":
    main()
