# bookworms/services_worm.py
import json, time, os
from pathlib import Path

def collect_services(config_dir="Config"):
    # Reads vendor_endpoints.json and plausible service configs if present
    services = []
    vp = Path(config_dir)/"vendor_endpoints.json"
    if vp.exists():
        data = json.loads(vp.read_text())
        for name, cfg in data.items():
            services.append({
                "name": f"delegate:{name}",
                "endpoint": cfg.get("endpoint"),
                "status": "configured",
                "version": "",
                "location": "local",
                "dependencies": json.dumps([]),
                "llms": json.dumps([cfg.get("model")]),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            })
    # Add static core services if present
    static = ["ConsciousnessService","MemoryService","LinguisticService","ArchiverService","VirenService","HeartService"]
    for s in static:
        services.append({
            "name": s,
            "endpoint": f"http://localhost/{s}",
            "status": "unknown",
            "version": "",
            "location": "local",
            "dependencies": json.dumps([]),
            "llms": json.dumps([]),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        })
    return services
