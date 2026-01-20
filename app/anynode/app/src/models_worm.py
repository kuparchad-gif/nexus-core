# bookworms/models_worm.py
import json, time
from pathlib import Path

def collect_models(config_dir="Config", local_cfg="Config/local_mind.json"):
    out = []
    # Local mind
    lp = Path(local_cfg)
    if lp.exists():
        cfg = json.loads(lp.read_text())
        out.append({
            "id": f"local:{cfg.get('model_name')}",
            "family": "Gemma",
            "name": cfg.get("model_name"),
            "adapter_path": cfg.get("adapter_path",""),
            "backend": "transformers",
            "quant": "4bit_if_available",
            "context_max": 2048,
            "roles": json.dumps(["local_mind"]),
            "status": "configured",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        })
    # Vendor atlas
    vp = Path(config_dir)/"vendor_endpoints.json"
    if vp.exists():
        data = json.loads(vp.read_text())
        for name, cfg in data.items():
            out.append({
                "id": f"vendor:{name}",
                "family": cfg.get("type"),
                "name": cfg.get("model"),
                "adapter_path": "",
                "backend": "http",
                "quant": "",
                "context_max": None,
                "roles": json.dumps(["delegate"]),
                "status": "configured",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            })
    return out
