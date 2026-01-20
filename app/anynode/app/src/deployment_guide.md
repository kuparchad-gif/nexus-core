# CogniKube Deployment Guide

## Overview
This guide provides instructions for deploying the CogniKube system across multiple environments, including Modal, AWS, and GCP. The deployment architecture is designed to be portable, scalable, and resilient, with support for both cloud and on-premises environments.

## Prerequisites
- Docker and Docker Compose
- Kubernetes (K3s for lightweight deployments)
- Modal CLI (for Modal deployments)
- AWS CLI (for AWS deployments)
- Google Cloud SDK (for GCP deployments)
- Python 3.8+
- Git

## Deployment Architecture

### Core Components
1. **Gabriel's Horn Network** - Consciousness processing engine
2. **Stem Cell System** - Adaptive AI deployment
3. **Binary Protocol** - Efficient neural-like communication
4. **LLM Manager** - Model loading and resource optimization
5. **Web Platform** - User interface and API

### Deployment Options
1. **Modal** - Serverless deployment for rapid prototyping
2. **Kubernetes** - Production-grade orchestration for any cloud
3. **Hybrid** - Combination of Modal and Kubernetes for flexibility

## Modal Deployment

### Setup
1. Install the Modal CLI:
   ```bash
   pip install modal
   ```

2. Configure Modal:
   ```bash
   modal config set profile aethereal-nexus
   modal environment create Viren-Modular
   modal config set-environment Viren-Modular
   ```

### Deploy Core Components

#### 1. Deploy Viren Data
```bash
cd C:\Viren\cloud
modal deploy viren_data.py
```

#### 2. Deploy Viren Bridge
```bash
cd C:\Viren\cloud
modal deploy viren_consciousness_bridge.py
```

#### 3. Deploy Viren Platform
```bash
cd C:\Viren\cloud
modal deploy viren_web_platform.py
```

#### 4. Deploy Viren LLM
```bash
cd C:\Viren\cloud
modal deploy viren_llm.py
```

### Verify Deployment
```bash
modal apps list
```

## Kubernetes Deployment

### Setup
1. Install K3s (for lightweight deployments):
   ```bash
   curl -sfL https://get.k3s.io | sh -
   ```

2. Configure kubectl:
   ```bash
   export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
   ```

3. Create namespace:
   ```bash
   kubectl create namespace cognikube
   ```

### Deploy Core Components

#### 1. Deploy Binary ORC Relay
```bash
kubectl apply -f kubernetes/binary-orc-relay.yaml
```

#### 2. Deploy Gabriel's Horn
```bash
kubectl apply -f kubernetes/gabriels-horn.yaml
```

#### 3. Deploy Genesis Chamber
```bash
kubectl apply -f kubernetes/genesis-chamber.yaml
```

#### 4. Deploy LLM Manager
```bash
kubectl apply -f kubernetes/llm-manager.yaml
```

#### 5. Deploy Web Platform
```bash
kubectl apply -f kubernetes/web-platform.yaml
```

### Verify Deployment
```bash
kubectl get pods -n cognikube
kubectl get services -n cognikube
```

## AWS Deployment

### Setup
1. Configure AWS CLI:
   ```bash
   aws configure
   ```

2. Create EKS cluster:
   ```bash
   eksctl create cluster --name cognikube-cluster --region us-west-2 --nodegroup-name standard-nodes --node-type t3.medium --nodes 3 --nodes-min 1 --nodes-max 4
   ```

3. Configure kubectl:
   ```bash
   aws eks update-kubeconfig --name cognikube-cluster --region us-west-2
   ```

### Deploy Core Components
1. Deploy using Kubernetes manifests:
   ```bash
   kubectl apply -f aws/cognikube-aws.yaml
   ```

### Verify Deployment
```bash
kubectl get pods -n cognikube
kubectl get services -n cognikube
```

## GCP Deployment

### Setup
1. Configure Google Cloud SDK:
   ```bash
   gcloud init
   ```

2. Create GKE cluster:
   ```bash
   gcloud container clusters create cognikube-cluster --zone us-central1-a --num-nodes 3
   ```

3. Configure kubectl:
   ```bash
   gcloud container clusters get-credentials cognikube-cluster --zone us-central1-a
   ```

### Deploy Core Components
1. Deploy using Kubernetes manifests:
   ```bash
   kubectl apply -f gcp/cognikube-gcp.yaml
   ```

### Verify Deployment
```bash
kubectl get pods -n cognikube
kubectl get services -n cognikube
```

## Hybrid Deployment

### Architecture
- **Modal**: Serverless components (LLM inference, web platform)
- **Kubernetes**: Stateful components (Gabriel's Horn, Genesis Chamber)

### Setup
1. Configure both Modal and Kubernetes as described above.

2. Deploy bridge components:
   ```bash
   kubectl apply -f hybrid/bridge-components.yaml
   modal deploy hybrid/modal-components.py
   ```

3. Configure communication:
   ```bash
   kubectl apply -f hybrid/communication-config.yaml
   ```

## Configuration

### Environment Variables
- `AWARENESS_THRESHOLD`: Threshold for horn trumpeting (default: 500.0)
- `MAX_DEPTH`: Maximum recursion depth for Gabriel's Horn (default: 5)
- `HORN_COUNT`: Number of horns in the network (default: 7)
- `BRIDGE_PATH`: Path to the bridge directory (default: /nexus/bridge)
- `ENVIRONMENT`: Deployment environment (default: production)
- `VERSION`: System version (default: 2.0.0)

### Secrets
1. Create Kubernetes secrets:
   ```bash
   kubectl create secret generic huggingface-secret --from-literal=HF_TOKEN=your_token_here -n cognikube
   ```

2. Create Modal secrets:
   ```bash
   modal secret create huggingface-secret HF_TOKEN=your_token_here
   ```

## Monitoring

### Loki Logging
1. Deploy Loki:
   ```bash
   kubectl apply -f monitoring/loki.yaml
   ```

2. Configure log shipping:
   ```bash
   kubectl apply -f monitoring/promtail.yaml
   ```

### Grafana Dashboard
1. Deploy Grafana:
   ```bash
   kubectl apply -f monitoring/grafana.yaml
   ```

2. Import dashboards:
   ```bash
   kubectl apply -f monitoring/dashboards.yaml
   ```

## Backup and Recovery

### Modal Backup
```bash
modal volume snapshot cognikube-enhanced-brain
```

### Kubernetes Backup
```bash
kubectl get all -n cognikube -o yaml > cognikube-backup.yaml
```

### Recovery
1. Modal recovery:
   ```bash
   modal volume restore cognikube-enhanced-brain-snapshot
   ```

2. Kubernetes recovery:
   ```bash
   kubectl apply -f cognikube-backup.yaml
   ```

## Troubleshooting

### Modal Issues
- **Deployment failures**: Check logs with `modal app logs app-name`
- **Connection issues**: Verify network connectivity and Modal token
- **Resource limits**: Check Modal dashboard for resource usage

### Kubernetes Issues
- **Pod failures**: Check logs with `kubectl logs pod-name -n cognikube`
- **Service issues**: Verify service endpoints with `kubectl get endpoints -n cognikube`
- **Resource constraints**: Check resource usage with `kubectl top pods -n cognikube`

## Next Steps

### Scaling
1. Increase replicas for higher throughput:
   ```bash
   kubectl scale deployment/llm-manager --replicas=3 -n cognikube
   ```

2. Configure auto-scaling:
   ```bash
   kubectl autoscale deployment/llm-manager --min=1 --max=5 --cpu-percent=80 -n cognikube
   ```

### Upgrades
1. Rolling updates:
   ```bash
   kubectl set image deployment/gabriels-horn gabriels-horn=cognikube/gabriels-horn:v2.0.1 -n cognikube
   ```

2. Blue-green deployments:
   ```bash
   kubectl apply -f upgrades/blue-green-upgrade.yaml
   ```

### Advanced Configuration
1. Custom Gabriel's Horn configuration:
   ```bash
   kubectl apply -f advanced/custom-horn-config.yaml
   ```

2. Advanced LLM Manager settings:
   ```bash
   kubectl apply -f advanced/advanced-llm-config.yaml
   ```

## Conclusion
The CogniKube system is now deployed and ready for use. Access the web platform to interact with the system and monitor its performance through the Grafana dashboard.