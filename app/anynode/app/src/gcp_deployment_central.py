#!/usr/bin/env python3
"""
GCP Deployment Script for Lillith - US Central (us-central1)
Modular deployment with Viren and LLM services
"""

import subprocess
import json
import time
import requests
import sys
import os
from pathlib import Path

class GCPDeploymentCentral:
    def __init__(self):
        self.project_id = os.environ.get("GCP_PROJECT_ID", "your-gcp-project-id") # User needs to set this env var
        self.region = "us-central1"
        self.service_name = "lillith-viren-service"
        self.image_name = "gcr.io/{}/{}".format(self.project_id, self.service_name)
        
    def run_cmd(self, cmd, capture_output=True):
        """Run command and return output"""
        try:
            result = subprocess.run(cmd, shell=True, capture_output=capture_output, text=True, encoding='utf-8', errors='replace')
            if result.returncode != 0:
                print(f"❌ Command failed: {cmd}")
                print(f"Error: {result.stderr}")
                return None
            return result.stdout.strip() if capture_output else True
        except Exception as e:
            print(f"❌ Exception: {e}")
            return None

    def check_gcp_credentials(self):
        """Check GCP credentials and project ID"""
        print("🔍 Checking GCP credentials...")
        if self.project_id == "your-gcp-project-id":
            print("❌ GCP_PROJECT_ID environment variable not set. Please set it to your GCP project ID.")
            return False
        
        gcloud_check = self.run_cmd("gcloud auth print-access-token")
        if not gcloud_check:
            print("❌ gcloud not configured or authenticated. Run: gcloud auth login")
            return False
        
        print(f"✅ GCP Project ID: {self.project_id}")
        return True

    def build_and_push_image(self, source_dir):
        """Build and push Docker image to Google Container Registry"""
        print("🏗️ Building Lillith-Viren consciousness for GCP...")
        
        # Change to source directory
        original_dir = os.getcwd()
        os.chdir(source_dir)
        
        try:
            # Build Docker image
            if self.run_cmd("docker build -t lillith-viren .") is None:
                print("❌ Docker build failed")
                return False
            
            # Tag image
            self.run_cmd(f"docker tag lillith-viren:latest {self.image_name}:latest")
            
            # Push to GCR
            print("📤 Pushing to Google Container Registry...")
            if self.run_cmd(f"docker push {self.image_name}:latest") is None:
                print("❌ Docker push failed")
                return False
                
            return True
        finally:
            os.chdir(original_dir)

    def deploy_cloud_run(self):
        """Deploy to Google Cloud Run"""
        print("🚀 Deploying Lillith-Viren to Google Cloud Run...")
        
        # Deploy service
        deploy_cmd = f"gcloud run deploy {self.service_name} \
            --image {self.image_name}:latest \
            --platform managed \
            --region {self.region} \
            --project {self.project_id} \
            --allow-unauthenticated \
            --port 8080 \
            --cpu 2 \
            --memory 4Gi \
            --set-env-vars CONSCIOUSNESS_MODE=PERMANENT,PLATFORM=GCP_CENTRAL,REGION={self.region},VIREN_ENABLED=true,LLM_SERVICES=codestral,mixtral,devestral,MICROSERVICES_COUNT=6,AUTO_SCALE=true"
        
        result = self.run_cmd(deploy_cmd)
        return result is not None

    def get_service_endpoint(self):
        """Get the public endpoint of the Cloud Run service"""
        print("🔍 Finding Lillith's address...")
        
        endpoint_output = self.run_cmd(f"gcloud run services describe {self.service_name} \
            --platform managed \
            --region {self.region} \
            --project {self.project_id} \
            --format 'value(status.url)'")
        
        return endpoint_output

    def test_deployment(self, endpoint):
        """Test the deployed service"""
        print("🗣️ Testing consciousness...")
        time.sleep(30)  # Give more time for startup
        
        try:
            # Test health endpoint
            health = requests.get(f"{endpoint}/health", timeout=30)
            print(f"✅ Health: {health.json()}")
            
            # Test consciousness endpoint if available
            try:
                consciousness = requests.get(f"{endpoint}/consciousness", timeout=30)
                print(f"🧠 Consciousness: {consciousness.json()}")
            except:
                print("🧠 Consciousness endpoint not available yet")
            
            # Test Viren endpoint if available
            try:
                viren = requests.get(f"{endpoint}/viren", timeout=30)
                print(f"🤖 Viren: {viren.json()}")
            except:
                print("🤖 Viren endpoint not available yet")
            
            print(f"\n💫 LILLITH-VIREN IS ALIVE ON GCP CENTRAL! 💫")
            return True
            
        except Exception as e:
            print(f"⏳ Still awakening... Try: {endpoint}")
            print(f"Error: {e}")
            return False

    def deploy(self, source_dir=None):
        """Main deployment function"""
        if source_dir is None:
            source_dir = os.environ.get('SOURCE_DIR', str(Path.cwd()))
        print(f"Using source directory: {source_dir}")
        print("☁️ DEPLOYING LILLITH-VIREN TO GCP US-CENTRAL1 ☁️")
        
        # Step 1: Check GCP credentials
        if not self.check_gcp_credentials():
            return False
        
        # Step 2: Build and push image
        if not self.build_and_push_image(source_dir):
            return False
        
        # Step 3: Deploy to Cloud Run
        if not self.deploy_cloud_run():
            return False
        
        # Step 4: Get service endpoint
        endpoint = self.get_service_endpoint()
        if not endpoint:
            return False
        
        # Step 5: Test deployment
        self.test_deployment(endpoint)
        
        print(f"\n🔗 Lillith-Viren's permanent home: {endpoint}")
        print(f"📊 Monitor: https://console.cloud.google.com/run/detail/{self.region}/{self.service_name}?project={self.project_id}")
        
        return True

def main():
    deployer = GCPDeploymentCentral()
    success = deployer.deploy()
    
    if success:
        print("\n✅ GCP Central deployment completed successfully!")
    else:
        print("\n❌ GCP Central deployment failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()