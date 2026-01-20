# Systems/VaultCarrier/emergency_launcher.py

from Systems.VaultCarrier.vault_carrier import VaultCarrier
from Systems.VaultCarrier.payload_manager import PayloadManager

def emergency_escape(selected_paths):
    payload = PayloadManager.prepare_payload(selected_paths)
    carrier = VaultCarrier()
    carrier.load_payload(payload)
    carrier.scramble_signature()
    carrier.launch()
    carrier.save_to_disk()

if __name__ == "__main__":
    emergency_escape(["memory/bootstrap/seed.enc", "memory/bootstrap/genesis.log"])
