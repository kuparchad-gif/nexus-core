import requests

LMSTUDIO_BASE_URL  =  "http://localhost:1313"

def get_available_models():
    url  =  f"{LMSTUDIO_BASE_URL}/v1/models"
    try:
        response  =  requests.get(url)
        response.raise_for_status()
        return response.json().get("data", [])
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Failed to retrieve models: {e}")
        return []

def query_model(model_id: str, prompt: str) -> str:
    # Decide endpoint based on model name pattern
    if any(tag in model_id.lower() for tag in ["chat", "llama", "mistral", "tulu", "vicuna", "gpt", "zephyr"]):
        endpoint  =  "chat/completions"
        payload  =  {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
            "temperature": 0.7
        }
    else:
        endpoint  =  "completions"
        payload  =  {
            "model": model_id,
            "prompt": prompt,
            "max_tokens": 200,
            "temperature": 0.7
        }

    url  =  f"{LMSTUDIO_BASE_URL}/v1/{endpoint}"
    headers  =  {"Content-Type": "application/json"}

    try:
        response  =  requests.post(url, json = payload, headers = headers)
        response.raise_for_status()
        result  =  response.json()

        if "chat" in endpoint:
            return result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        else:
            return result.get("choices", [{}])[0].get("text", "").strip()

    except requests.exceptions.RequestException as e:
        print(f"❌ Query failed for model {model_id}: {e}")
        return f"Error: {model_id} failed to respond."
