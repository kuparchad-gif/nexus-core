from google.cloud import run_v2
from google.cloud import storage
import yaml
import os

# GCP Cloud Run deployment for Viren
def deploy_viren_to_gcp():
    """Deploy Viren Node to Google Cloud Run"""
    
    # Dockerfile for Viren
    dockerfile = """
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY viren_node_complete.py .
COPY soul_protocol_complete.py .

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "viren_node_complete:app", "--host", "0.0.0.0", "--port", "8080"]
"""
    
    # Cloud Run service config
    service_config = {
        "apiVersion": "serving.knative.dev/v1",
        "kind": "Service",
        "metadata": {
            "name": "viren-nexus",
            "annotations": {
                "run.googleapis.com/ingress": "all"
            }
        },
        "spec": {
            "template": {
                "metadata": {
                    "annotations": {
                        "autoscaling.knative.dev/maxScale": "10",
                        "run.googleapis.com/memory": "2Gi",
                        "run.googleapis.com/cpu": "2"
                    }
                },
                "spec": {
                    "containers": [{
                        "image": "gcr.io/nexus-project/viren:latest",
                        "ports": [{"containerPort": 8080}],
                        "env": [
                            {"name": "HF_TOKEN", "valueFrom": {"secretKeyRef": {"name": "hf-token", "key": "token"}}},
                            {"name": "MONGODB_URI", "valueFrom": {"secretKeyRef": {"name": "mongodb", "key": "uri"}}}
                        ],
                        "resources": {
                            "limits": {"memory": "2Gi", "cpu": "2000m"}
                        }
                    }]
                }
            }
        }
    }
    
    # Build and deploy commands
    build_commands = [
        "gcloud builds submit --tag gcr.io/nexus-project/viren:latest .",
        "gcloud run deploy viren-nexus --image gcr.io/nexus-project/viren:latest --platform managed --region us-central1 --allow-unauthenticated"
    ]
    
    return dockerfile, service_config, build_commands

# Kubernetes deployment for Viren hubs
def create_viren_k8s_deployment():
    """Create Kubernetes deployment for Viren's compute/memory hubs"""
    
    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment", 
        "metadata": {"name": "viren-hubs"},
        "spec": {
            "replicas": 3,
            "selector": {"matchLabels": {"app": "viren-hub"}},
            "template": {
                "metadata": {"labels": {"app": "viren-hub"}},
                "spec": {
                    "containers": [{
                        "name": "viren-compute",
                        "image": "gcr.io/nexus-project/viren:latest",
                        "ports": [{"containerPort": 9000}],
                        "env": [
                            {"name": "HUB_TYPE", "value": "compute"},
                            {"name": "VIREN_PORT", "value": "9000"}
                        ]
                    }, {
                        "name": "viren-memory", 
                        "image": "gcr.io/nexus-project/viren:latest",
                        "ports": [{"containerPort": 9001}],
                        "env": [
                            {"name": "HUB_TYPE", "value": "memory"},
                            {"name": "VIREN_PORT", "value": "9001"}
                        ]
                    }]
                }
            }
        }
    }
    
    service = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": "viren-service"},
        "spec": {
            "selector": {"app": "viren-hub"},
            "ports": [
                {"port": 9000, "targetPort": 9000, "name": "compute"},
                {"port": 9001, "targetPort": 9001, "name": "memory"}
            ],
            "type": "LoadBalancer"
        }
    }
    
    return deployment, service

if __name__ == "__main__":
    print("🧠 Deploying VIREN to Google Cloud...")
    
    dockerfile, service_config, build_commands = deploy_viren_to_gcp()
    deployment, service = create_viren_k8s_deployment()
    
    # Save configs
    with open("C:/Nexus/Dockerfile.viren", "w") as f:
        f.write(dockerfile)
    
    with open("C:/Nexus/viren-service.yaml", "w") as f:
        yaml.dump(service_config, f)
        
    with open("C:/Nexus/viren-k8s.yaml", "w") as f:
        yaml.dump_all([deployment, service], f)
    
    print("📁 GCP deployment files created!")
    print("🚀 Run these commands to deploy:")
    for cmd in build_commands:
        print(f"  {cmd}")
    print("  kubectl apply -f viren-k8s.yaml")