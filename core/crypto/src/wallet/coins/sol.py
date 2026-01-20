from __future__ import annotations
from ..core.networks import rpc_for
from ..core.validator import validate_address
from solders.pubkey import Pubkey
from solana.rpc.api import Client

def get_sol_balance(address: str, network: dict) -> float:
    if not validate_address(address, network["type"], network.get("address_prefix", "")):
        raise ValueError(f"Invalid {network['name']} address: {address}")
    rpc = rpc_for(network)
    if not rpc:
        return 0.0
    try:
        client = Client(rpc)
        balance = client.get_balance(Pubkey.from_string(address)).value
        return balance / 10**network.get("decimals", 9)
    except Exception as e:
        print(f"Failed to fetch SOL balance: {e}")
        return 0.0