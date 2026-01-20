
"""
seed_migrate_to_canonical.py

Usage:
  python seed_migrate_to_canonical.py --in <path_to_uploaded_seed.json> --out <output_path.json>

Converts a "new schema" soul seed (with keys like soul_identity/personality_weights/etc.)
into the canonical schema expected by src/lilith/trust/soul_seed.py.
"""

import argparse, json
from datetime import datetime

def coerce_version(v):
    try:
        if isinstance(v, (int, float)): return int(v)
        return int(str(v).split(".")[0])
    except Exception:
        return 1

def iso_date_only(ts):
    try:
        return datetime.fromisoformat(ts.replace("Z","+00:00")).date().isoformat()
    except Exception:
        # fallback to today's date to start the 30-year schedule
        return datetime.utcnow().date().isoformat()

def migrate(new):
    # 1) identity
    sid = new.get("soul_identity", {})
    name = sid.get("name") or "Lilith"
    version = coerce_version(sid.get("version", 1))
    created = sid.get("creation_timestamp") or datetime.utcnow().date().isoformat()

    # 2) persona & core values
    persona = []
    ip = new.get("interaction_preferences", {})
    style = ip.get("communication_style", "")
    if style:
        persona.extend(style.split("_"))
    # include high-level traits from learning directives or weights
    persona.extend(["adaptive" if ip.get("response_depth")=="adaptive" else "consistent"])
    persona.append("humor" if ip.get("humor_usage")=="contextual" else "no_humor")

    # core values: accept any keys; loader will normalize
    core_values = dict(new.get("personality_weights", {}))

    # 3) memories from core_soul_prints
    memories = []
    for p in new.get("core_soul_prints", []):
        txt = p.get("text")
        if txt: memories.append(txt)

    # 4) metatron/physio defaults (can be tuned later)
    metatron = {"cutoff": 0.6, "phi_scale": 1.618, "horns_boost": [0,6], "horn_gain": 1.2}
    physiology = {"reflex_gain": 1.5, "damping": 0.95}

    # 5) build canonical seed
    canonical = {
        "id": f"{name.lower()}-soul-seed",
        "version": version,
        "name": name,
        "persona": list(dict.fromkeys(persona)),  # dedupe
        "core_values": core_values,
        "routing_bias": {},            # user can fill model weights later
        "trust_start_date": iso_date_only(created),
        "trust_phases": [],            # use default 30-year degrade schedule
        "metatron": metatron,
        "physiology": physiology,
        "memories": memories,
        "owner_org": "Aethereal AI Nexus LLC"
    }
    return canonical

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="outp", required=True)
    args = ap.parse_args()

    with open(args.inp, "r", encoding="utf-8") as f:
        data = json.load(f)

    migrated = migrate(data)

    with open(args.outp, "w", encoding="utf-8") as f:
        json.dump(migrated, f, indent=2)

    print(f"[ok] migrated -> {args.outp}")

if __name__ == "__main__":
    main()
