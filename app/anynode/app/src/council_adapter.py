import math, json, time, hashlib, requests, os
from typing import Dict, List
from twilio.rest import Client

class CouncilAdapter:
    def __init__(self, weights: Dict[str,float], redlines: List[str], mcp_config: Dict):
        self.weights = weights
        self.redlines = set(redlines)
        self.mcp_config = mcp_config

    def aggregate(self, proposals: Dict[str, dict]) -> dict:
        for who, p in proposals.items():
            for tag in p.get("tags", []):
                if tag in self.redlines:
                    return {"decision":"blocked", "by":"redline", "tag":tag, "chosen": who}
        best = None; best_w = -1.0
        for who, p in proposals.items():
            score = float(p.get("score", 0.0))
            w = self.weights.get(who, 0.0) * score
            if w > best_w:
                best_w = w; best = (who, p)
        return {"decision":"approved", "chosen": best[0], "proposal": best[1], "weight": best_w}

    def try_mcp_request(self, endpoint: str, payload: dict) -> dict:
        for server_name, server in self.mcp_config["mcpServers"].items():
            if not server.get("enabled", True) or not server.get("url"):
                continue
            try:
                response = requests.post(f"{server['url']}/{endpoint}", json=payload, timeout=5)
                response.raise_for_status()
                return {"status": "success", "server": server_name, "response": response.json()}
            except Exception as e:
                self.notify_chad(f"MCP server {server_name} failed: {str(e)}")
        return {"status": "error", "message": "All MCP servers failed"}

    def notify_chad(self, message: str):
        client = Client(os.getenv("TWILIO_SID", "SK763698d08943c64a5beeb0bf29cdeb3a"), os.getenv("TWILIO_AUTH_TOKEN", ""))
        client.messages.create(body=message, from_="+18666123982", to="+17246126323")