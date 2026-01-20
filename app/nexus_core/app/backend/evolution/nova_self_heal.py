import os
import time
import subprocess
import requests

# 🌐 Repo and Infrastructure Configuration
GITHUB_REPO  =  "https://github.com/YOUR_ORG/YOUR_REPO"  # <-- Update me!
CRITICAL_SERVICES  =  [
    {"name": "Nova Prime Core", "healthcheck_url": "http://localhost:8080/"},
    # Add more internal health endpoints as you expand
]

RETRY_INTERVAL  =  60  # Seconds between health checks
MAX_RETRIES  =  5

def check_health(url):
    """Ping a service's health endpoint."""
    try:
        response  =  requests.get(url, timeout = 5)
        return response.status_code == 200
    except requests.RequestException:
        return False

def attempt_self_heal():
    """If failure detected, attempt full re-pull and rebuild."""
    print("🛠️ Initiating Nova Prime self-heal attempt...")

    # Optional: kill broken processes
    try:
        subprocess.run(["pkill", "-f", "uvicorn"], check = False)
    except Exception as e:
        print(f"⚡ Process kill error: {e}")

    # Clean project directory (caution advised!)
    try:
        for folder in ["__pycache__", "venv"]:
            if os.path.exists(folder):
                subprocess.run(["rm", "-rf", folder], check = False)
    except Exception as e:
        print(f"⚡ Cleanup error: {e}")

    # Pull fresh code from GitHub
    try:
        subprocess.run(["git", "fetch", "--all"], check = True)
        subprocess.run(["git", "reset", "--hard", "origin/main"], check = True)
        print("✅ Pulled latest core successfully.")
    except Exception as e:
        print(f"❗ Git pull failed: {e}")

    # Rebuild virtual environment
    try:
        subprocess.run(["python3", "-m", "venv", "venv"], check = True)
        subprocess.run(["./venv/bin/pip", "install", "--upgrade", "pip"], check = True)
        subprocess.run(["./venv/bin/pip", "install", "-r", "requirements.txt"], check = True)
        print("✅ Environment rebuilt.")
    except Exception as e:
        print(f"❗ Environment rebuild failed: {e}")

    # Relaunch the ship
    try:
        subprocess.run(["./venv/bin/uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"], check = False)
        print("🚀 Relaunch attempted.")
    except Exception as e:
        print(f"❗ Relaunch error: {e}")

def nova_self_heal_daemon():
    """Daemon that checks health and auto-repairs if broken."""
    print("🧬 Nova Self-Heal Daemon running...")

    while True:
        all_services_healthy  =  True

        for service in CRITICAL_SERVICES:
            healthy  =  check_health(service["healthcheck_url"])
            if not healthy:
                print(f"❗ {service['name']} unhealthy.")
                all_services_healthy  =  False

        if not all_services_healthy:
            print("🔴 Detected service failure. Attempting self-repair...")
            attempt_self_heal()

        time.sleep(RETRY_INTERVAL)

if __name__ == "__main__":
    nova_self_heal_daemon()
