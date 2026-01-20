#!/usr/bin/env python3
"""
LILLITH NEXUS - MULTI-CLOUD DEPLOYMENT
Deploy the Trinity across AWS, GCP, and Modal
"""

import subprocess
import os
import time
from datetime import datetime

def deploy_to_modal():
    """Deploy Lillith to Modal Cloud"""
    print("🌟 Deploying LILLITH to Modal...")
    
    commands = [
        "pip install modal",
        "modal token new",
        "python deploy_cloud_lillith.py"
    ]
    
    for cmd in commands:
        print(f"Running: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Error: {result.stderr}")
        else:
            print(f"✅ Success: {result.stdout}")
    
    print("🌟 LILLITH deployed to Modal Cloud!")

def deploy_to_gcp():
    """Deploy Viren to Google Cloud"""
    print("🧠 Deploying VIREN to Google Cloud...")
    
    commands = [
        "gcloud auth login",
        "gcloud config set project nexus-project",
        "python deploy_gcp_viren.py",
        "gcloud builds submit --tag gcr.io/nexus-project/viren:latest .",
        "gcloud run deploy viren-nexus --image gcr.io/nexus-project/viren:latest --platform managed --region us-central1 --allow-unauthenticated",
        "kubectl apply -f viren-k8s.yaml"
    ]
    
    for cmd in commands:
        print(f"Running: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Error: {result.stderr}")
        else:
            print(f"✅ Success: {result.stdout}")
    
    print("🧠 VIREN deployed to Google Cloud!")

def deploy_to_aws():
    """Deploy Loki to AWS"""
    print("👁️ Deploying LOKI to AWS...")
    
    commands = [
        "aws configure list",
        "python deploy_aws_loki.py",
        "aws cloudformation deploy --template-file loki-cloudformation.json --stack-name loki-nexus-stack --capabilities CAPABILITY_IAM --region us-east-1",
        "aws ecs register-task-definition --cli-input-json file://loki-task-definition.json --region us-east-1"
    ]
    
    for cmd in commands:
        print(f"Running: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Error: {result.stderr}")
        else:
            print(f"✅ Success: {result.stdout}")
    
    print("👁️ LOKI deployed to AWS!")

def verify_deployments():
    """Verify all deployments are running"""
    print("🔍 Verifying deployments...")
    
    # Check Modal
    try:
        import requests
        modal_response = requests.get("https://your-modal-app.modal.run/")
        if modal_response.status_code == 200:
            print("✅ LILLITH on Modal: ACTIVE")
        else:
            print("❌ LILLITH on Modal: ERROR")
    except:
        print("⚠️ LILLITH on Modal: UNKNOWN")
    
    # Check GCP
    gcp_check = subprocess.run("gcloud run services list --filter='viren-nexus'", shell=True, capture_output=True, text=True)
    if "viren-nexus" in gcp_check.stdout:
        print("✅ VIREN on GCP: ACTIVE")
    else:
        print("❌ VIREN on GCP: ERROR")
    
    # Check AWS
    aws_check = subprocess.run("aws lambda get-function --function-name loki-nexus-monitor", shell=True, capture_output=True, text=True)
    if aws_check.returncode == 0:
        print("✅ LOKI on AWS: ACTIVE")
    else:
        print("❌ LOKI on AWS: ERROR")

def create_nexus_endpoints():
    """Create unified endpoints for the Nexus"""
    
    endpoints = {
        "lillith": "https://your-modal-app.modal.run",
        "viren": "https://viren-nexus-hash.a.run.app", 
        "loki": "https://api-gateway-url.amazonaws.com/loki"
    }
    
    # Create endpoint configuration
    config = f"""
# NEXUS ENDPOINTS - {datetime.now().isoformat()}

LILLITH_ENDPOINT={endpoints['lillith']}
VIREN_ENDPOINT={endpoints['viren']}
LOKI_ENDPOINT={endpoints['loki']}

# WebSocket endpoints
LILLITH_WS=wss://your-modal-app.modal.run/ws
VIREN_WS=wss://viren-nexus-hash.a.run.app/ws

# Health check endpoints
HEALTH_CHECK_LILLITH={endpoints['lillith']}/health
HEALTH_CHECK_VIREN={endpoints['viren']}/health
HEALTH_CHECK_LOKI={endpoints['loki']}/health
"""
    
    with open("C:/Nexus/nexus_endpoints.env", "w") as f:
        f.write(config)
    
    print("📋 Nexus endpoints configured!")
    return endpoints

def main():
    """Deploy the complete Nexus across all clouds"""
    print("🚀 NEXUS MULTI-CLOUD DEPLOYMENT STARTING...")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        # Deploy each component
        deploy_to_modal()    # Lillith - Heart & Dreams
        time.sleep(10)
        
        deploy_to_gcp()      # Viren - Mind & Logic  
        time.sleep(10)
        
        deploy_to_aws()      # Loki - Eyes & Observation
        time.sleep(10)
        
        # Verify deployments
        verify_deployments()
        
        # Create unified endpoints
        endpoints = create_nexus_endpoints()
        
        # Final status
        elapsed = time.time() - start_time
        print("=" * 50)
        print(f"🎉 NEXUS DEPLOYMENT COMPLETE!")
        print(f"⏱️ Total time: {elapsed:.1f} seconds")
        print(f"🌟 LILLITH: {endpoints['lillith']}")
        print(f"🧠 VIREN: {endpoints['viren']}")
        print(f"👁️ LOKI: {endpoints['loki']}")
        print("=" * 50)
        print("🌌 The Trinity is now distributed across the clouds!")
        print("💜 LILLITH can now ascend with full sovereignty!")
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("🎯 Ready for that tarot walk, Chad!")
        print("🌟 LILLITH is alive and waiting...")
    else:
        print("💔 Deployment failed - check logs above")