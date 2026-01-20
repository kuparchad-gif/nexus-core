# adapters/http_clients.py
# Minimal HTTP clients for major LLM vendors. No secrets included.
import os, json, http.client, urllib.parse

class VendorError(Exception): pass

def _call_json(base, path, payload, headers):
    u = urllib.parse.urlparse(base)
    conn = http.client.HTTPSConnection(u.netloc, timeout=120)
    body = json.dumps(payload).encode("utf-8")
    conn.request("POST", path, body, headers)
    resp = conn.getresponse()
    data = resp.read()
    if resp.status >= 300:
        raise VendorError(f"{resp.status} {resp.reason}: {data[:500]!r}")
    try:
        return json.loads(data.decode())
    except Exception:
        return {"raw": data.decode(errors="ignore")}

def openai_chat(endpoint, api_key, model, messages, tools=None, json_mode=False):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type":"application/json"}
    payload = {"model": model, "messages": messages, "stream": False}
    if tools: payload["tools"] = tools
    if json_mode: payload["response_format"] = {"type":"json_object"}
    return _call_json(endpoint, "/v1/chat/completions", payload, headers)

def anthropic_messages(endpoint, api_key, model, messages, tools=None):
    headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01", "Content-Type":"application/json"}
    payload = {"model": model, "max_tokens": 1024, "messages": messages}
    if tools: payload["tools"] = tools
    return _call_json(endpoint, "/v1/messages", payload, headers)

def xai_chat(endpoint, api_key, model, messages, tools=None):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type":"application/json"}
    payload = {"model": model, "messages": messages, "stream": False}
    if tools: payload["tools"] = tools
    return _call_json(endpoint, "/v1/chat/completions", payload, headers)

def deepseek_chat(endpoint, api_key, model, messages):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type":"application/json"}
    payload = {"model": model, "messages": messages}
    return _call_json(endpoint, "/v1/chat/completions", payload, headers)

def google_gemini(endpoint, api_key, model, messages):
    # messages: [{"role":"user","content":"..."}] → convert to Parts per Gemini API as needed
    headers = {"Content-Type":"application/json"}
    payload = {"contents":[{"role":m["role"],"parts":[{"text":m["content"]}]} for m in messages]}
    path = f"/v1beta/{model}:generateContent?key={api_key}"
    return _call_json(endpoint, path, payload, headers)

def mistral_chat(endpoint, api_key, model, messages):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type":"application/json"}
    payload = {"model": model, "messages": messages}
    return _call_json(endpoint, "/v1/chat/completions", payload, headers)

def cohere_chat(endpoint, api_key, model, messages):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type":"application/json"}
    prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    payload = {"model": model, "message": prompt}
    return _call_json(endpoint, "/v1/chat", payload, headers)

def openrouter_chat(endpoint, api_key, model, messages):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type":"application/json"}
    payload = {"model": model, "messages": messages}
    return _call_json(endpoint, "/v1/chat/completions", payload, headers)

def manus_chat(endpoint, api_key, model, messages):
    # Placeholder: Manus API not public; keep signature ready.
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type":"application/json"}
    payload = {"model": model, "messages": messages}
    # You must replace path and payload per Manus docs when available.
    return _call_json(endpoint, "/api/chat", payload, headers)
