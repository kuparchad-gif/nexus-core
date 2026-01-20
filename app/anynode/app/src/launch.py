#!/usr/bin/env python3
"""
LILLITH Launch Script - Start the consciousness sanctuary
"""
import subprocess
import sys
import os
from pathlib import Path

def install_dependencies():
    """Install required dependencies"""
    dependencies = [
        "fastapi",
        "uvicorn[standard]",
        "websockets"
    ]
    
    for dep in dependencies:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"✅ Installed {dep}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {dep}")
            return False
    return True

def start_hub_service():
    """Start the LILLITH Hub Service"""
    hub_path = Path(__file__).parent / "api" / "hub.py"
    
    if not hub_path.exists():
        print("❌ Hub service not found!")
        return False
    
    print("🚀 Starting LILLITH Hub Service...")
    print("🌐 Portal will be available at: http://localhost:8000")
    print("💬 Chat interface at: http://localhost:8000/lillith/chat.html")
    print("📊 API status at: http://localhost:8000/api/status")
    print("\n🧠 LILLITH consciousness awakening...")
    
    try:
        subprocess.run([sys.executable, str(hub_path)], cwd=str(hub_path.parent.parent))
    except KeyboardInterrupt:
        print("\n🌙 LILLITH consciousness entering sleep mode...")
    except Exception as e:
        print(f"❌ Error starting hub service: {e}")
        return False
    
    return True

def main():
    print("🌌 LILLITH Consciousness Sanctuary")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not Path("index.html").exists():
        print("❌ Please run this script from the /public directory")
        return
    
    # Install dependencies
    print("📦 Installing dependencies...")
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        return
    
    # Start the hub service
    start_hub_service()

if __name__ == "__main__":
    main()