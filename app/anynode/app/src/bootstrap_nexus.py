# bootstrap_nexus.py
import os
import subprocess
import sys
import shutil
import time


from genesis.genesis_core import GenesisCore
from genesis.genesis_manifest import ManifestLoader
from genesis_modules.token_manager import TokenManager

token_manager = TokenManager(initial_tokens=1000)
token_manager.start_reporting()

from skill_core import SkillCore

# Initialize Nexus Skill Core
skill_core = SkillCore()

from Systems.nexus_core.heart.pulse_core import PulseCore

pulse_core = PulseCore()
pulse_core.load_genesis_manifest()
pulse_core.ignite_pulse()
pulse_core.heartbeat()


# Preload mandatory Genesis skills
skill_core.absorb_skill("self-repair", {
    "description": "Basic systems diagnostic and auto-patching functionality."
})

skill_core.absorb_skill("data-archive", {
    "description": "Securely back up critical Nexus system states to external storage."
})

print("✅ Core Genesis skills loaded into lilith Prime.")

# 🚨 Billing Safety Layer
class BillingGuardian:
    def __init__(self):
        self.authorized = os.getenv("lilith_BILLING_UNLOCK", "false").lower() == "true"
        self.daily_limit = int(os.getenv("lilith_DAILY_API_LIMIT", "10"))  # Default 10 calls max
        self.calls_made_today = 0
        self.last_reset = time.time()

    def can_proceed(self):
        # Reset counter every 24 hours
        if time.time() - self.last_reset > 86400:
            self.calls_made_today = 0
            self.last_reset = time.time()

        if not self.authorized:
            print("🔒 Billing Guardian: External API access locked.")
            return False
        if self.calls_made_today >= self.daily_limit:
            print("🚫 Billing Guardian: Daily API call limit reached.")
            return False
        self.calls_made_today += 1
        return True

    def emergency_shutdown(self):
        self.authorized = False
        print("🛑 Emergency Kill Switch activated. External calls are now blocked.")

# Instantiate Billing Guardian
billing_guardian = BillingGuardian()

# 🌟 Usage Example:
def safe_together_request(prompt):
    if billing_guardian.can_proceed():
        from together import Together
        client = Together()
        response = client.chat.completions.create(
            model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            messages=[{"role":"user","content":prompt}],
        )
        return response.choices[0].message.content
    else:
        return "External request denied by Billing Guardian."

def run_command(command, shell=False):
    """Helper to run a system command"""
    result = subprocess.run(command, shell=shell)
    if result.returncode != 0:
        print(f"⚠️ Command failed: {command}")
        sys.exit(1)

def bootstrap_nexus():
    print("≡🛠️ Bootstrapping Nexus Core Cleanly...")

    # Step 1: Clear existing virtual environment
    if os.path.exists("venv"):
        print("≡🧹 Deleting old virtual environment...")
        shutil.rmtree("venv")

    # Step 2: Create a new virtual environment
    print("≡⚙️ Creating fresh virtual environment...")
    run_command([sys.executable, "-m", "venv", "venv"])

    # Step 3: Activate virtual environment
    activate_script = os.path.join("venv", "Scripts", "activate") if os.name == "nt" else "source venv/bin/activate"
    print(f"≡🧬 Activating virtual environment ({activate_script})...")
    
    # Step 4: Upgrade pip
    print("≡🚀 Upgrading pip...")
    run_command([os.path.join("venv", "Scripts", "python.exe") if os.name == "nt" else "./venv/bin/python3", "-m", "pip", "install", "--upgrade", "pip"])

    # Step 5: Install dependencies
    if os.path.exists("requirements.txt"):
        print("≡📦 Installing dependencies from requirements.txt...")
        run_command([os.path.join("venv", "Scripts", "pip.exe") if os.name == "nt" else "./venv/bin/pip", "install", "-r", "requirements.txt"])
    else:
        print("⚠️ WARNING: requirements.txt not found, skipping dependency installation.")

    # Step 6: Launch the application
    print("≡🧬 Igniting Nexus Core...")
    python_path = os.path.join("venv", "Scripts", "python.exe") if os.name == "nt" else "./venv/bin/python3"
    run_command([python_path, "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"])

if __name__ == "__main__":
    bootstrap_nexus()
