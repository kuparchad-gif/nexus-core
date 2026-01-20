# Minimal OpenAI-compatible chat adapter (stdlib urllib)
import json
from urllib import request, error
class OpenAICompat:
    def __init__(self, base_url: str, api_key: str = "", model: str = "gpt-4o-mini"):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
    def chat(self, prompt: str, temperature: float = 0.2, system: str = "You are a helpful assistant.") -> str:
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role":"system","content":system},
                {"role":"user","content":prompt}
            ],
            "temperature": temperature,
            "stream": False
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = request.Request(url, data=data, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=120) as resp:
                body = resp.read()
                obj = json.loads(body.decode("utf-8", errors="ignore"))
                return obj["choices"][0]["message"]["content"]
        except error.HTTPError as e:
            msg = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"HTTPError {e.code}: {msg}") from None
        except Exception as e:
            raise RuntimeError(f"Adapter error: {e}") from None
