import re

def validate_address(addr: str, network_type: str, prefix: str = "") -> bool:
    if network_type == "evm":
        return addr.startswith("0x") and len(addr) == 42
    if network_type == "tron":
        return addr.startswith("T") and len(addr) in (34,)
    if network_type == "utxo":
        return addr.startswith("tb1") or addr.startswith("bc1") or addr.startswith("1") or addr.startswith("3")
    if network_type == "solana":
        return len(addr) > 30  # naive base58 length check
    return False
