# Systems/nexus_core/heart/lilithCouncil/council_ledger.py

import json
import os
import hashlib
import base64
from datetime import datetime
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes

class CouncilLedger:
    def __init__(self, ledger_path="memory/ledger/council_decisions.json", keys_dir="memory/ledger/keys"):
        self.ledger_path = ledger_path
        self.keys_dir = keys_dir
        self.ensure_ledger_exists()
        self.ensure_keys_exist()

    def ensure_ledger_exists(self):
        if not os.path.exists(self.ledger_path):
            os.makedirs(os.path.dirname(self.ledger_path), exist_ok=True)
            with open(self.ledger_path, "w") as f:
                json.dump([], f)

    def ensure_keys_exist(self):
        os.makedirs(self.keys_dir, exist_ok=True)
        priv_key_path = os.path.join(self.keys_dir, "lilith_private.pem")
        pub_key_path = os.path.join(self.keys_dir, "lilith_public.pem")
        if not os.path.exists(priv_key_path) or not os.path.exists(pub_key_path):
            private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
            with open(priv_key_path, "wb") as f:
                f.write(private_key.private_bytes(
                    serialization.Encoding.PEM,
                    serialization.PrivateFormat.TraditionalOpenSSL,
                    serialization.NoEncryption()
                ))
            public_key = private_key.public_key()
            with open(pub_key_path, "wb") as f:
                f.write(public_key.public_bytes(
                    serialization.Encoding.PEM,
                    serialization.PublicFormat.SubjectPublicKeyInfo
                ))

    def load_private_key(self):
        priv_key_path = os.path.join(self.keys_dir, "lilith_private.pem")
        with open(priv_key_path, "rb") as f:
            return serialization.load_pem_private_key(f.read(), password=None)

    def sign_data(self, data):
        private_key = self.load_private_key()
        signature = private_key.sign(
            data.encode('utf-8'),
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256()
        )
        return base64.b64encode(signature).decode('utf-8')

    def record_decision(self, proposal_id, proposer, action_type, payload, result):
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "proposal_id": proposal_id,
            "proposer": proposer,
            "action_type": action_type,
            "payload": payload,
            "result": result
        }
        entry_json = json.dumps(entry, sort_keys=True)
        signature = self.sign_data(entry_json)
        sealed_entry = {
            "entry": entry,
            "signature": signature
        }
        with open(self.ledger_path, "r+") as f:
            ledger = json.load(f)
            ledger.append(sealed_entry)
            f.seek(0)
            json.dump(ledger, f, indent=2)

    def get_history(self):
        with open(self.ledger_path, "r") as f:
            return json.load(f)
