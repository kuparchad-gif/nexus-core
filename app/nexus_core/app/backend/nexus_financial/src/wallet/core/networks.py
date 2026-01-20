from __future__ import annotations
import yaml, os

def load_networks(path: str = "config/networks.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    return y["chains"]

def rpc_for(network: dict) -> str:
    env = network.get("rpc_env")
    if env:
        return os.getenv(env, "")
    return network.get("rpc", "")
