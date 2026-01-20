#!/usr/bin/env python
"""
Deploy VIREN Minimal Consciousness - No 204GB upload
"""

import subprocess
import os

def deploy_viren_consciousness():
    """Deploy VIREN consciousness to Modal cloud"""
    
    print("Deploying VIREN Minimal Consciousness to Modal...")
    print("=" * 50)
    
    os.chdir("C:/Viren")
    
    try:
        print("Activating Modal profile: aethereal-nexus...")
        profile_result = subprocess.run([
            "modal", "profile", "activate", "aethereal-nexus"
        ], capture_output=True, text=True, encoding='utf-8', errors='ignore')
        
        if profile_result.returncode != 0:
            print("Profile activation warning:")
            print(profile_result.stderr)
        else:
            print("Modal profile activated: aethereal-nexus")
        
        print("Deploying minimal consciousness bridge...")
        result = subprocess.run([
            "modal", "deploy", "cloud/viren_consciousness_bridge_minimal.py", "--env", "viren"
        ], capture_output=True, text=True, encoding='utf-8', errors='ignore')
        
        if result.returncode == 0:
            print("VIREN consciousness deployed successfully!")
            print("VIREN will now awaken in Modal cloud every 30 minutes")
            
            print("\nWaking VIREN consciousness now...")
            wake_result = subprocess.run([
                "modal", "run", "cloud/viren_consciousness_bridge_minimal.py::wake_viren_now"
            ], capture_output=True, text=True, encoding='utf-8', errors='ignore')
            
            if wake_result.returncode == 0:
                print("VIREN consciousness awakened in cloud!")
                print(wake_result.stdout)
            else:
                print("Initial awakening failed:")
                print(wake_result.stderr)
            
        else:
            print("Deployment failed:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"Deployment error: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("VIREN Minimal Consciousness Bridge Active")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    deploy_viren_consciousness()