
# C:\Engineers\eden_engineering\scripts\llama2_baidu_api.py
# Baidu LLM access script with full logging to chat_logs

import requests
import json
from pathlib import Path
from datetime import datetime

LOG_DIR = Path(__file__).parent.parent / "logs" / "chat_logs"
LOG_FILE = LOG_DIR / "llama2_baidu.log"

LOG_DIR.mkdir(parents=True, exist_ok=True)

def get_baiduqianfan_access_token():
    # Dummy token logic – replace with real token retrieval
    return "YOUR_ACCESS_TOKEN"

def log_event(event):
    timestamp = datetime.now().isoformat()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} :: {event}\n")

def query_baidu_llm(prompt):
    token = get_baiduqianfan_access_token()
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/llama_2_7b"
    headers = {"Content-Type": "application/json"}
    payload = {
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(f"{url}?access_token={token}", headers=headers, data=json.dumps(payload))
        log_event(f"REQUEST: {json.dumps(payload)}")
        if response.status_code == 200:
            result = response.json()
            log_event(f"RESPONSE: {json.dumps(result)}")
            return result.get("result", "")
        else:
            log_event(f"ERROR {response.status_code}: {response.text}")
            return None
    except Exception as e:
        log_event(f"EXCEPTION: {str(e)}")
        return None

if __name__ == "__main__":
    print("USING: llama2_baidu_api")
    reply = query_baidu_llm("What is the capital of France?")
    print("Response:", reply)
