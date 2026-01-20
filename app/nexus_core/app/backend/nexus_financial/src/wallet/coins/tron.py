from __future__ import annotations
from ..core.networks import rpc_for
from ..core.validator import validate_address
from tronpy import Tron

def get_trx_balance(address: str, network: dict) -> float:
    if not validate_address(address, network["type"], network.get("address_prefix", "")):
        raise ValueError(f"Invalid {network['name']} address: {address}")
    rpc = rpc_for(network)
    if not rpc:
        return 0.0
    try:
        client = Tron(network="nile" if "nile" in network["key"] else "mainnet")
        balance = client.get_account_balance(address)
        return balance / 10**network.get("decimals", 6)
    except Exception as e:
        print(f"Failed to fetch TRX balance: {e}")
        return 0.0