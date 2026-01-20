# adapters/delegates.py
import os, json
from pathlib import Path
from .http_clients import (
    openai_chat, anthropic_messages, xai_chat, deepseek_chat, google_gemini,
    mistral_chat, cohere_chat, openrouter_chat, manus_chat, VendorError
)

def load_vendor_config(path: str):
    return json.loads(Path(path).read_text())

def call_delegate(name: str, cfg: dict, messages: list):
    """
    Vendor delegate multiplexer. Supports:
      - type == 'local'  : echo stub, no endpoint/model needed
      - type == 'http'   : POST to endpoint with payload
      - type == 'openai' : call OpenAI chat completions (if configured)
    """
    t = (cfg.get("type") or "").lower()

    # --- local stub: no network, no model ---
    if t == "local":
        return {
            "vendor": name,
            "type": "local",
            "output": {
                "role": "assistant",
                "content": f"[local:{name}] " + (messages[-1]["content"] if messages else "")
            }
        }

    # --- http generic ---
    if t == "http":
        import requests, json
        endpoint = cfg.get("endpoint")
        if not endpoint:
            return {"error": f"vendor '{name}' missing 'endpoint' for type=http"}
        timeout = cfg.get("timeout", 15)
        payload = {
            "model": cfg.get("model"),
            "messages": messages,
            "params": cfg.get("params", {})
        }
        try:
            r = requests.post(endpoint, json=payload, timeout=timeout)
            r.raise_for_status()
            return {"vendor": name, "type": "http", "output": r.json()}
        except Exception as e:
            return {"vendor": name, "type": "http", "error": str(e)}

    # --- openai example (optional) ---
    if t == "openai":
        try:
            from openai import OpenAI
            model = cfg.get("model", "gpt-4o-mini")
            client = OpenAI(api_key=cfg.get("api_key"))
            resp = client.chat.completions.create(model=model, messages=messages)
            return {"vendor": name, "type": "openai", "output": resp.dict()}
        except Exception as e:
            return {"vendor": name, "type": "openai", "error": str(e)}

    return {"error": f"vendor '{name}' has unknown or missing type"}