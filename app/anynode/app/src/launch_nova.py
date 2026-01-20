# launch_nova.py
import subprocess
import os
import sys

def activate_virtualenv():
    """Activate virtual environment if it exists."""
    venv_activate = os.path.join('venv', 'Scripts', 'activate')
    if not os.path.exists('venv'):
        print("⚙️ No venv detected. Creating virtual environment...")
        subprocess.run([sys.executable, '-m', 'venv', 'venv'])
    else:
        print("✅ Virtual environment detected.")

def install_requirements():
    """Install dependencies."""
    print("📦 Installing/Verifying dependencies...")
    subprocess.run([os.path.join('venv', 'Scripts', 'pip'), 'install', '-r', 'requirements.txt'])

def launch_nova():
    """Launch Nova Prime Engine."""
    print("🚀 Launching Nova Prime Core...")
    subprocess.run([os.path.join('venv', 'Scripts', 'python'), 'bootstrap_nexus.py'])

if __name__ == "__main__":
    print("🔋 Bootstrapping Nova Prime Deployment...")
    activate_virtualenv()
    install_requirements()
    launch_nova()
