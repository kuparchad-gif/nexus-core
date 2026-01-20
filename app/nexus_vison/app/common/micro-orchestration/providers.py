
import os, json, requests

class LLMProvider:
    def classify(self, prompt: str) -> str:
        raise NotImplementedError

def _prefer_content(msg: dict) -> str:
    # Prefer assistant content; fall back to reasoning_content for "thinking" models
    return (msg.get("content") or msg.get("reasoning_content") or "").strip()

class LMStudioProvider(LLMProvider):
    def __init__(self, url: str, model: str):
        # Accept both "http://host:port" and ".../v1/chat/completions" inputs
        url  =  url.rstrip("/")
        if not url.endswith("/v1/chat/completions"):
            url  =  url + "/v1/chat/completions"
        self.url  =  url
        self.model  =  model

    def classify(self, prompt: str) -> str:
        payload  =  {
            "model": self.model,
            "messages": [{"role":"user","content": prompt}],
            "temperature": 0.0,
            "max_tokens": int(os.getenv("LM_MAX_TOKENS", "-1")),  # match user's working curl
            "stream": False
        }
        try:
            r  =  requests.post(self.url, json = payload, timeout = 60)
            r.raise_for_status()
        except requests.HTTPError as e:
            body  =  (r.text or "")[:400]
            print(f"[lmstudio] HTTP {r.status_code}: {body}")
            raise
        data  =  r.json()
        msg  =  data["choices"][0]["message"]
        return _prefer_content(msg)

class OllamaProvider(LLMProvider):
    def __init__(self, url: str, model: str):
        self.url  =  url.rstrip("/")
        self.model  =  model
    def classify(self, prompt: str) -> str:
        r  =  requests.post(f"{self.url}/api/generate", json = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }, timeout = 60)
        r.raise_for_status()
        data  =  r.json()
        return data.get("response","")

class VLLMProvider(LLMProvider):
    def __init__(self, url: str, model: str):
        url  =  url.rstrip("/")
        if not url.endswith("/v1/chat/completions"):
            url  =  url + "/v1/chat/completions"
        self.url  =  url
        self.model  =  model
    def classify(self, prompt: str) -> str:
        r  =  requests.post(self.url, json = {
            "model": self.model,
            "messages": [{"role":"user","content": prompt}],
            "temperature": 0.0,
            "max_tokens": int(os.getenv("LM_MAX_TOKENS", "256")),
            "stream": False
        }, timeout = 60)
        r.raise_for_status()
        data  =  r.json()
        msg  =  data["choices"][0]["message"]
        return _prefer_content(msg)

def make_provider(env: dict):
    provider  =  (env.get("PROVIDER") or "none").lower()
    if provider == "lmstudio":
        return LMStudioProvider(env.get("LMSTUDIO_URL","http://localhost:1234"), env.get("LMSTUDIO_MODEL","qwen/qwen3-4b-thinking-2507"))
    if provider == "ollama":
        return OllamaProvider(env.get("OLLAMA_URL","http://localhost:11434"), env.get("OLLAMA_MODEL","qwen3:4b"))
    if provider == "vllm":
        return VLLMProvider(env.get("VLLM_URL","http://localhost:8000"), env.get("VLLM_MODEL","qwen/qwen3-4b-instruct"))
    return None
