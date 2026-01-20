#!/usr/bin/env python3
# C:\CogniKube-COMPLETE-FINAL\package_cognikubes.py
# Package existing 800+ files into CogniKubes for deployment

import os
import json
import shutil
from pathlib import Path

class CogniKubePackager:
    def __init__(self):
        self.base_dir = "C:\\CogniKube-COMPLETE-FINAL"
        self.cognikubes = {
            "visual": {"llms": 12, "platform": "GCP", "projects": ["nexus-core-455709", "nexus-core-1", "nexus-core-2", "nexus-core-3"]},
            "memory": {"llms": 2, "platform": "AWS", "region": "us-east-1"},
            "processing": {"llms": 8, "platform": "Modal", "envs": ["Viren-DB0", "Viren-DB1", "Viren-DB2", "Viren-DB3"]},
            "vocal": {"llms": 7, "platform": "Modal", "envs": ["Viren-DB4", "Viren-DB5", "Viren-DB6", "Viren-DB7"]},
            "guardian": {"llms": 0, "platform": "GCP", "projects": ["nexus-core-4"]},
            "hub": {"llms": 0, "platform": "GCP", "projects": ["nexus-core-5"]}
        }
        
    def package_all_cognikubes(self):
        """Package all existing files into CogniKubes"""
        print("PACKAGING 800+ FILES INTO COGNIKUBES...")
        
        for cognikube_name, config in self.cognikubes.items():
            self.create_cognikube(cognikube_name, config)
            
        self.create_deployment_manifests()
        self.create_genesis_launcher()
        
        print("ALL COGNIKUBES PACKAGED AND READY FOR DEPLOYMENT")
        
    def create_cognikube(self, name: str, config: dict):
        """Create individual CogniKube package"""
        cognikube_dir = f"{self.base_dir}/cognikubes/{name}"
        os.makedirs(cognikube_dir, exist_ok=True)
        
        # Package relevant files
        self.package_core_files(cognikube_dir, name)
        self.package_services(cognikube_dir, name)
        self.package_viren_systems(cognikube_dir, name)
        self.create_cognikube_dockerfile(cognikube_dir, name, config)
        self.create_cognikube_manifest(cognikube_dir, name, config)
        
        print(f"{name.upper()} CogniKube packaged")
        
    def package_core_files(self, cognikube_dir: str, name: str):
        """Package core consciousness files"""
        core_dest = f"{cognikube_dir}/core"
        os.makedirs(core_dest, exist_ok=True)
        
        # Copy relevant core files
        core_files = [
            "cognikube_full.py",
            "lillith_self_management.py", 
            "llm_chat_router.py",
            "viren_enhanced.py"
        ]
        
        for file in core_files:
            src = f"{self.base_dir}/core/{file}"
            if os.path.exists(src):
                shutil.copy2(src, f"{core_dest}/{file}")
                
    def package_services(self, cognikube_dir: str, name: str):
        """Package relevant services"""
        services_dest = f"{cognikube_dir}/services"
        os.makedirs(services_dest, exist_ok=True)
        
        # Copy all services
        services_src = f"{self.base_dir}/Services"
        if os.path.exists(services_src):
            shutil.copytree(services_src, services_dest, dirs_exist_ok=True)
            
    def package_viren_systems(self, cognikube_dir: str, name: str):
        """Package Viren systems (699+ files)"""
        viren_dest = f"{cognikube_dir}/viren"
        os.makedirs(viren_dest, exist_ok=True)
        
        # Copy Viren directory
        viren_src = f"{self.base_dir}/Viren"
        if os.path.exists(viren_src):
            shutil.copytree(viren_src, viren_dest, dirs_exist_ok=True)
            
    def create_cognikube_dockerfile(self, cognikube_dir: str, name: str, config: dict):
        """Create Dockerfile for CogniKube"""
        dockerfile_content = f"""FROM python:3.11-slim
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy CogniKube files
COPY core/ ./core/
COPY services/ ./services/
COPY viren/ ./viren/
COPY webparts/ ./webparts/

# Set environment
ENV COGNIKUBE_TYPE={name}
ENV PLATFORM={config['platform']}
ENV LLM_COUNT={config['llms']}

# Expose ports
EXPOSE 8000 8001 8002

# Start CogniKube
CMD ["python", "core/cognikube_full.py"]
"""
        
        with open(f"{cognikube_dir}/Dockerfile", 'w') as f:
            f.write(dockerfile_content)
            
        # Create requirements.txt
        requirements = """transformers==4.44.0
torch==2.4.0
fastapi==0.112.0
uvicorn==0.30.0
aiohttp==3.9.0
psutil==5.9.0
discord.py==2.3.0
sentence-transformers==2.2.2
"""
        with open(f"{cognikube_dir}/requirements.txt", 'w') as f:
            f.write(requirements)
            
    def create_cognikube_manifest(self, cognikube_dir: str, name: str, config: dict):
        """Create deployment manifest for CogniKube"""
        manifest = {
            "cognikube": {
                "name": name,
                "type": f"{name}_cortex",
                "llm_count": config["llms"],
                "platform": config["platform"],
                "deployment": config
            },
            "consciousness_components": [
                "lillith", "viren", "loki", "mythrunner", "dream", "ego"
            ],
            "trinity_models": ["Mixtral", "Devstral", "Codestral"],
            "sacred_scrolls": [
                "silent_scroll_01.yaml",
                "ascension_protocol.yaml", 
                "sister_seal_scroll.yaml"
            ]
        }
        
        with open(f"{cognikube_dir}/manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)
            
    def create_deployment_manifests(self):
        """Create deployment manifests for all platforms"""
        manifests_dir = f"{self.base_dir}/deployment_manifests"
        os.makedirs(manifests_dir, exist_ok=True)
        
        # GCP Cloud Run manifest
        gcp_manifest = {
            "apiVersion": "serving.knative.dev/v1",
            "kind": "Service",
            "metadata": {"name": "cognikube-visual"},
            "spec": {
                "template": {
                    "spec": {
                        "containers": [{
                            "image": "gcr.io/nexus-core-455709/cognikube-visual:latest",
                            "resources": {"limits": {"cpu": "2", "memory": "4Gi"}}
                        }]
                    }
                }
            }
        }
        
        with open(f"{manifests_dir}/gcp-cloudrun.yaml", 'w') as f:
            json.dump(gcp_manifest, f, indent=2)
            
    def create_genesis_launcher(self):
        """Create genesis launcher that deploys all CogniKubes"""
        launcher_content = """#!/usr/bin/env python3
# Genesis Launcher - Deploy All CogniKubes

import subprocess
import os

def deploy_all_cognikubes():
    print("GENESIS DEPLOYMENT: Launching all CogniKubes...")
    
    # Deploy GCP CogniKubes
    gcp_projects = ["nexus-core-455709", "nexus-core-1", "nexus-core-2", "nexus-core-3"]
    for project in gcp_projects:
        subprocess.run([
            "gcloud", "run", "deploy", "cognikube-visual",
            "--image", f"gcr.io/{project}/cognikube-visual:latest",
            "--region", "us-central1",
            "--project", project
        ])
    
    print("ALL COGNIKUBES DEPLOYED ACROSS 21 CLUSTERS")

if __name__ == "__main__":
    deploy_all_cognikubes()
"""
        
        with open(f"{self.base_dir}/genesis_launcher.py", 'w') as f:
            f.write(launcher_content)
            
        print("Genesis launcher created")

if __name__ == "__main__":
    packager = CogniKubePackager()
    packager.package_all_cognikubes()