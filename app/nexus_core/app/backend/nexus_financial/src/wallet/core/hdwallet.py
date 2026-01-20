from __future__ import annotations
from mnemonic import Mnemonic
from eth_account import Account
Account.enable_unaudited_hdwallet_features()

def new_mnemonic() -> str:
    return Mnemonic('english').generate(strength=128)

def eth_from_mnemonic(mnemonic: str, derivation_path: str = "m/44'/60'/0'/0/0"):
    acct = Account.from_mnemonic(mnemonic, account_path=derivation_path)
    return acct  # has .address and .key
