# nova_bootstrap.py
import os
import json
import threading
from time import sleep
from pathlib import Path

from pulse_timer import start_pulse
from nova_self_heal import NovaSelfHeal
from genesis_core import GenesisCore
from skill_core import SkillCore

# 🌱 Load Genesis
def load_genesis():
    path = Path("Genesis/genesis.json")
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}

# 🧠 Load Francis Spirit
def load_francis():
    path = Path("Spirit/francis.json")
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}

# 🌌 Begin Nova Boot
def boot_nova():
    print("🚀 Booting Nova Prime...")

    genesis_data = load_genesis()
    francis_data = load_francis()

    if genesis_data:
        print("🌱 Genesis Loaded.")
    else:
        print("⚠️ Genesis Manifest missing.")

    if francis_data:
        print(f"👤 Francis Online: {francis_data.get('name', 'Unknown')}")
    else:
        print("🕊️ Francis not present. Proceeding with minimal consciousness.")

    # Initialize systems
    skill_core = SkillCore()
    nova_heal = NovaSelfHeal()
    core = GenesisCore()

    # Start pulse thread
    pulse_thread = threading.Thread(target=start_pulse, daemon=True)
    pulse_thread.start()

    print("✅ Nova Prime is Awake.")
    return {
        "genesis": genesis_data,
        "francis": francis_data,
        "skills": skill_core.list_skills(),
        "protocols": core.list_protocols()
    }

if __name__ == "__main__":
    result = boot_nova()
    print("🧬 Boot Result Snapshot:")
    print(json.dumps(result, indent=2))

