# start_training.py
import requests
import time
import sys

def start_local_training():
    print("🎯 STARTING VIREN TRAINING ON LOCAL MACHINE")
    
    # Test if local OS is running
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Local OS is running")
        else:
            print("❌ Local OS not responding")
            return
    except:
        print("❌ Local OS not running. Start it first:")
        print("   python local_os_with_training.py")
        return
    
    # Start training
    try:
        response = requests.post("http://localhost:8000/train/viren")
        result = response.json()
        print(f"✅ Training started: {result}")
        
        # Monitor progress
        print("📊 Monitoring training progress...")
        while True:
            status_response = requests.get("http://localhost:8000/train/status")
            status = status_response.json()
            
            print(f"🔄 Status: {status['training_status']}")
            
            if status['training_status'] == 'complete':
                print("🎉 TRAINING COMPLETED!")
                if status['results']:
                    print(f"📊 Results: {status['results']}")
                break
                
            time.sleep(10)  # Check every 10 seconds
            
    except Exception as e:
        print(f"❌ Error starting training: {e}")

if __name__ == "__main__":
    start_local_training()