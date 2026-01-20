# C:\CogniKube-COMPLETE-FINAL\retry_deployments.py
# Retry failed deployments with enhanced diagnostics

import subprocess
import requests
import time
import os

def check_health(url):
    """Check if endpoint is healthy"""
    try:
        response = requests.get(f"{url}/health", timeout=10)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}

def deploy_layer(layer_file, environment, retries=3, delay=5):
    """Deploy layer with retries and detailed logging"""
    print(f"Deploying {layer_file} to Viren-DB{environment}...")
    
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    
    for attempt in range(retries):
        try:
            result = subprocess.run([
                "modal", "deploy", layer_file, "-e", f"Viren-DB{environment}"
            ], capture_output=True, text=True, cwd="C:\\CogniKube-COMPLETE-FINAL", env=env)
            
            if result.returncode == 0:
                print(f"SUCCESS: {layer_file} deployed to Viren-DB{environment}")
                return True, result.stdout
            else:
                print(f"FAILED: Attempt {attempt+1}/{retries}")
                print(f"STDERR: {result.stderr}")
                print(f"STDOUT: {result.stdout}")
                
        except Exception as e:
            print(f"ERROR: Attempt {attempt+1}/{retries} - {e}")
        
        if attempt < retries - 1:
            print(f"Retrying in {delay} seconds...")
            time.sleep(delay)
    
    return False, "Max retries exceeded"

def get_endpoint_url(layer_file, environment):
    """Generate expected endpoint URL"""
    if "bert" in layer_file:
        return f"https://aethereal-nexus-viren-db{environment}--bert-layer-bert-processor.modal.run"
    elif "orchestrator_layer" in layer_file:
        return f"https://aethereal-nexus-viren-db{environment}--orchestrator-layer-orchestrator.modal.run"
    elif "service_orchestrator" in layer_file:
        return f"https://aethereal-nexus-viren-db{environment}--service-orchestrator-layer-service-orchestrator.modal.run"
    return None

def main():
    print("RETRY DEPLOYMENTS - FIXING VIREN-DB2 AND DB3")
    print("=" * 60)
    
    layers = [
        "bert_layer.py",
        "orchestrator_layer.py", 
        "service_orchestrator_layer.py"
    ]
    
    # Focus on failed environments
    environments = [2, 3]
    
    for env in environments:
        print(f"\n{'='*20} VIREN-DB{env} {'='*20}")
        
        for layer in layers:
            print(f"\n--- {layer} ---")
            
            # Try deployment
            success, output = deploy_layer(layer, env)
            
            if success:
                # Test health endpoint
                endpoint_url = get_endpoint_url(layer, env)
                if endpoint_url:
                    print(f"Testing health: {endpoint_url}")
                    health_ok, health_data = check_health(endpoint_url)
                    print(f"Health Status: {'HEALTHY' if health_ok else 'UNHEALTHY'}")
                    print(f"Health Data: {health_data}")
                else:
                    print("Could not determine endpoint URL")
            else:
                print(f"Deployment failed: {output}")
            
            time.sleep(3)  # Brief pause between layers
    
    print(f"\n{'='*60}")
    print("RETRY DEPLOYMENT COMPLETE")
    print("Check output above for specific failures")

if __name__ == "__main__":
    main()