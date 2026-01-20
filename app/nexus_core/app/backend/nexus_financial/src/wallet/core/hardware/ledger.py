# src/wallet/hardware/ledger.py
class LedgerSigner:
    def __init__(self, derivation_path="m/44'/60'/0'/0/0"):
        self.derivation_path = derivation_path
    
    def sign_transaction(self, tx_data, network):
        # Use ledgerblue or similar library
        with getLedger() as ledger:
            signed = ledger.sign_transaction(tx_data, self.derivation_path)
            return signed