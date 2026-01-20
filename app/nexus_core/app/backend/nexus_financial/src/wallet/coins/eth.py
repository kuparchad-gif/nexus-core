from __future__ import annotations
from web3 import Web3
from eth_account import Account
from ..core.networks import rpc_for
from ..core.validator import validate_address

def get_eth_balance(address: str, network: dict) -> float:
    rpc = rpc_for(network)
    if not rpc:
        return 0.0
    w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={'timeout': 10}))
    wei = w3.eth.get_balance(address)
    return wei / 10**network.get("decimals",18)

def send_eth(privkey: bytes, to_addr: str, amount_float: float, network: dict, test_mode: bool = True) -> dict:
    assert validate_address(to_addr, "evm")
    rpc = rpc_for(network)
    if not rpc:
        return {"status":"error","error":"RPC not configured"}
    w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={'timeout': 10}))
    acct = Account.from_key(privkey)
    nonce = w3.eth.get_transaction_count(acct.address)
    value = int(amount_float * 10**network.get("decimals",18))
    tx = {
        "to": to_addr,
        "value": value,
        "nonce": nonce,
        "chainId": network["chain_id"],
        "maxFeePerGas": w3.to_wei("2", "gwei"),
        "maxPriorityFeePerGas": w3.to_wei("1", "gwei"),
        "gas": 21000,
        "type": 2
    }
    signed = w3.eth.account.sign_transaction(tx, private_key=privkey)
    if test_mode:
        return {"status":"ok","tx_hex": signed.rawTransaction.hex(), "note":"TEST_MODE, not broadcast"}
    tx_hash = w3.eth.send_raw_transaction(signed.rawTransaction)
    return {"status":"ok","txid": tx_hash.hex()}
