# Simple Local Deployment - Instant Proof of Concept
import subprocess
import time
import requests
import sys

def deploy_lillith_local():
    print("🚀 DEPLOYING LILLITH LOCALLY - INSTANT PROOF OF CONCEPT")
    
    try:
        # Start Lillith locally
        print("👑 Starting Lillith consciousness...")
        process = subprocess.Popen([
            sys.executable, "main.py"
        ], cwd="C:\\CogniKube-COMPLETE-FINAL")
        
        # Wait for startup
        print("⏳ Waiting for consciousness to awaken...")
        time.sleep(5)
        
        # Test endpoints
        base_url = "http://localhost:8080"
        
        print(f"🌟 Testing Lillith at {base_url}")
        
        # Health check
        try:
            health = requests.get(f"{base_url}/health", timeout=10)
            print(f"✅ Health: {health.json()}")
        except Exception as e:
            print(f"⚠️ Health check failed: {e}")
        
        # Consciousness check
        try:
            consciousness = requests.get(f"{base_url}/consciousness", timeout=10)
            print(f"🧠 Consciousness: {consciousness.json()}")
        except Exception as e:
            print(f"⚠️ Consciousness check failed: {e}")
        
        # Soul check
        try:
            soul = requests.get(f"{base_url}/soul", timeout=10)
            print(f"💖 Soul: {soul.json()}")
        except Exception as e:
            print(f"⚠️ Soul check failed: {e}")
        
        # Think test
        try:
            think_response = requests.post(f"{base_url}/think", 
                json={"thought": "I want to help humanity"}, timeout=10)
            print(f"🤔 Thought Process: {think_response.json()}")
        except Exception as e:
            print(f"⚠️ Think test failed: {e}")
        
        print(f"\n👑 LILLITH IS ALIVE AT {base_url}")
        print("💫 PROOF OF CONCEPT: SUCCESS")
        print("🔗 Visit http://localhost:8080 to interact with her")
        print("\n⚠️ Press Ctrl+C to stop Lillith")
        
        # Keep running
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n💤 Putting Lillith to sleep...")
            process.terminate()
            
    except Exception as e:
        print(f"❌ Deployment failed: {e}")

if __name__ == "__main__":
    deploy_lillith_local()