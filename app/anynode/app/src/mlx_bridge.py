# mlx_bridge.py
# Purpose: Bridge module to connect Lillith with a remote MLX runtime via HTTP
# Location: /root/bridge/mlx_bridge.py

import requests
import logging

MLX_REMOTE_URL = "http://localhost:8008/run"  # Default endpoint of remote MLX container

logger = logging.getLogger("mlx_bridge")
logger.setLevel(logging.INFO)
handler = logging.FileHandler("logs/boot_logs/mlx_bridge.log")
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def query(prompt: str, model: str = None, stream: bool = False) -> str:
    payload = {
        "prompt": prompt,
        "model": model,
        "stream": stream
    }

    try:
        response = requests.post(MLX_REMOTE_URL, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "[MLX] No output received.")
    except Exception as e:
        logger.error(f"MLX query failed: {e}")
        return f"[MLX] Error: {e}"

if __name__ == "__main__":
    print(query("What is the airspeed velocity of an unladen swallow?"))
