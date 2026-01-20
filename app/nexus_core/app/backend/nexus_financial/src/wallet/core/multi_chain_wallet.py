```python
from bitcoinlib.services.services import Service
from bitcoinlib.transactions import Transaction
from ..core.networks import rpc_for
from ..core.validator import validate_address
from ..utils.logging_config import logger

def create_transaction(privkey: str, to_addr: str, amount: float, network: dict) -> Transaction:
    try:
        if not validate_address(to_addr, network["type"], network.get("address_prefix", "")):
            raise ValueError(f"Invalid {network['name']} address: {to_addr}")
        svc = Service(network=network["key"], provider=rpc_for(network))
        tx = Transaction(network=network["key"])
        amount_satoshi = int(amount * 10**network.get("decimals", 8))
        tx.add_output(amount_satoshi, to_addr)
        tx.sign(privkey)
        logger.info(f"BTC transaction created for {amount} to {to_addr}")
        return tx
    except Exception as e:
        logger.error(f"Failed to create BTC transaction: {e}")
        raise

def broadcast_transaction(tx: Transaction, rpc_url: str) -> dict:
    try:
        svc = Service(provider=rpc_url)
        txid = svc.sendrawtransaction(tx.raw_hex())
        logger.info(f"BTC transaction broadcast: txid={txid}")
        return {"status": "ok", "txid": txid}
    except Exception as e:
        logger.error(f"Failed to broadcast BTC transaction: {e}")
        return {"status": "error", "error": str(e)}

def send_btc(privkey: str, to_addr: str, amount: float, network: dict, test_mode: bool = True) -> dict:
    rpc_url = rpc_for(network)
    if not rpc_url:
        logger.error("RPC URL not configured for BTC network")
        return {"status": "error", "error": "RPC not configured"}
    
    tx = create_transaction(privkey, to_addr, amount, network)
    if test_mode:
        logger.info("BTC send in test mode; transaction not broadcast")
        return {"status": "test", "tx_hex": tx.raw_hex()}
    return broadcast_transaction(tx, rpc_url)
```