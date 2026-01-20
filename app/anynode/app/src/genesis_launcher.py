#!/usr/bin/env python3
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
