           # genesis_manifest_updater.py
from genesis.genesis_manifest import GENESIS_MANIFEST

def update_manifest(key, value):
    if key in GENESIS_MANIFEST["NOVA_CORE"]:
        GENESIS_MANIFEST["NOVA_CORE"][key] = value
        print(f"✅ Manifest updated: {key} -> {value}")
    else:
        print(f"❌ Key {key} not found in manifest.")
                                                                                                                                                                                                                                                                                                                                              