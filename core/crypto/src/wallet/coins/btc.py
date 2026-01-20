from __future__ import annotations
from ..core.networks import rpc_for
from ..core.validator import validate_address
from bitcoinlib.services.services import Service

def get_btc_balance(address: str, network: dict) -> float:
    if not validate_address(address, network["type"], network.get("address_prefix", "")):
        raise ValueError(f"Invalid {network['name']} address: {address}")
    rpc = rpc_for(network)
    if not rpc:
        return 0.0
    try:
        svc = Service(network=network["key"], provider=rpc)
        balance = svc.getbalance(address)
        return balance / 10**network.get("decimals", 8)
    except Exception as e:
        print(f"Failed to fetch BTC balance: {e}")
        return 0.0