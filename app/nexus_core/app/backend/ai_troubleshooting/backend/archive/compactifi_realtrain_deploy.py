# deploy_real_training.py
import subprocess
import sys
import time

def deploy_real_system():
    print("🎯 DEPLOYING REAL TRAINING SYSTEM")
    
    # Test PyTorch availability
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ CUDA: {torch.cuda.is_available()}")
    except ImportError:
        print("❌ PyTorch not available")
        return False

    # Test transformers
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("✅ Transformers available")
    except ImportError:
        print("❌ Transformers not available")
        return False

    print("🚀 Deploying to Modal...")
    try:
        result = subprocess.run([
            sys.executable, "-m", "modal", "deploy", "CognikubeSupermeshOS_RealTraining.py"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ REAL TRAINING OS DEPLOYED SUCCESSFULLY!")
            print("🔗 Access at: https://cognikube-os-realtraining.modal.run")
            return True
        else:
            print(f"❌ Deployment failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Deployment error: {e}")
        return False

if __name__ == "__main__":
    success = deploy_real_system()
    if success:
        print("\n🎉 REAL TRAINING SYSTEM READY!")
        print("Use: POST /train/viren to start Viren training")
        print("Or connect via WebSocket and send {'action': 'start_training'}")
    else:
        print("\n💥 Deployment failed - check dependencies")