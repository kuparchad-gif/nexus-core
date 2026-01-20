# CogniKube Deployment Guide

## Pre-Deployment Checklist

### 1. Install Prerequisites
```powershell
# Install Docker Desktop
winget install Docker.DockerDesktop

# Install Google Cloud SDK
winget install Google.CloudSDK

# Install AWS CLI
winget install Amazon.AWSCLI

# Install Modal CLI
pip install modal

# Install PowerShell 7+
winget install Microsoft.PowerShell
```

### 2. Configure Credentials

#### Update `config\secrets.json`:
```json
{
  "aws": {
    "account": "YOUR_AWS_ACCOUNT_ID",
    "access_key": "YOUR_ACCESS_KEY", 
    "secret_key": "YOUR_SECRET_KEY"
  },
  "modal": {
    "token": "YOUR_MODAL_TOKEN"
  },
  "secrets": {
    "huggingface_token": "YOUR_HF_TOKEN"
  }
}
```

#### Authenticate Cloud Services:
```powershell
# Google Cloud
gcloud auth login
gcloud auth application-default login

# AWS
aws configure

# Modal
modal token new
```

## Deployment Process

### 1. Extract Package
```powershell
# Extract to C:\CogniKube
cd C:\CogniKube
```

### 2. Update Configuration
```powershell
# Edit credentials
notepad config\secrets.json

# Verify services configuration
notepad config\services.json
```

### 3. Deploy CogniKube
```powershell
# Run main deployment
.\deploy.ps1

# Or deploy to specific platform
.\deploy.ps1 -Platform gcp
.\deploy.ps1 -Platform aws  
.\deploy.ps1 -Platform modal
```

### 4. Verify Deployment
```powershell
# Check all services
.\scripts\verify-deploy.ps1

# Check specific platform
gcloud run services list --project=nexus-core-455709
aws ecs list-task-definitions
modal app list
```

## Architecture Overview

### Free Tier Distribution
- **GCP Cloud Run**: 3 Visual Cortex CogniKubes (12 LLMs) - $0
- **AWS Fargate**: 1 Memory Cortex CogniKube (2 LLMs) - $0
- **Modal**: 2 Processing CogniKubes (15 LLMs) - ~$30/month

### Resource Allocation
Each LLM container gets:
- **CPU**: 2 cores dedicated
- **Memory**: 4GB dedicated  
- **Isolation**: Complete container isolation
- **Networking**: ANYNODE mesh connectivity

## Service Details

### Visual Cortex 1 (GCP)
- **LLMs**: LLaVA-Video-7B, Intel/dpt-large, google/vit-base, stabilityai/stable-fast-3d
- **Resources**: 8 cores, 16GB total
- **Platform**: GCP Cloud Run (Free Tier)

### Visual Cortex 2 (GCP)  
- **LLMs**: google/vit-in21k, ashawkey/LGM, facebook/sam-vit-huge, ETH-CVG/lightglue
- **Resources**: 8 cores, 16GB total
- **Platform**: GCP Cloud Run (Free Tier)

### Visual Cortex 3 (GCP)
- **LLMs**: calcuis/wan-gguf, facebook/vjepa2, prompthero/openjourney, deepseek-ai/Janus-1.3B
- **Resources**: 8 cores, 16GB total  
- **Platform**: GCP Cloud Run (Free Tier)

### Memory Cortex (AWS)
- **LLMs**: Qwen/Qwen2.5-Omni-3B, deepseek-ai/Janus-1.3B
- **Resources**: 4 cores, 8GB total
- **Platform**: AWS Fargate (Free Tier)

### Processing Cortex (Modal)
- **LLMs**: whisper-large-v3, all-MiniLM-L6-v2, phi-2, bart-large-cnn, bart-large-mnli, roberta-base-squad2, bert-base-NER, distilbert-sst-2
- **Resources**: 16 cores, 32GB total
- **Platform**: Modal (~$15/month)

### Vocal Cortex (Modal)
- **LLMs**: Dia-1.6B, musicgen-small, whisper-large-v3, xcodec2, ast-audioset, Janus-1.3B, Qwen2.5-Omni-3B  
- **Resources**: 14 cores, 28GB total
- **Platform**: Modal (~$15/month)

## Monitoring & Management

### Health Checks
```powershell
# Check service health
curl http://SERVICE_URL/health

# Get detailed status  
curl http://SERVICE_URL/status
```

### Scaling
```powershell
# Scale GCP service
gcloud run services update SERVICE_NAME --max-instances=10

# Scale AWS service
aws ecs update-service --desired-count 2

# Scale Modal function
modal run modal-deploy.py::scale_service
```

### Logs
```powershell
# GCP logs
gcloud logs read "resource.type=cloud_run_revision"

# AWS logs
aws logs describe-log-groups

# Modal logs  
modal logs cognikube-nested
```

## Troubleshooting

### Common Issues

#### Docker Build Fails
```powershell
# Clean Docker cache
docker system prune -a

# Rebuild images
.\scripts\build-images.ps1
```

#### GCP Deployment Fails
```powershell
# Check quotas
gcloud compute project-info describe --project=PROJECT_ID

# Enable APIs
gcloud services enable run.googleapis.com
```

#### AWS Deployment Fails
```powershell
# Check ECS service
aws ecs describe-services --cluster default --services SERVICE_NAME

# Check task definition
aws ecs describe-task-definition --task-definition SERVICE_NAME
```

#### Modal Deployment Fails
```powershell
# Check Modal status
modal app list

# Redeploy
modal deploy templates\modal-deploy.py
```

### Resource Limits

#### GCP Free Tier Limits
- **CPU**: 8 vCPU max per service
- **Memory**: 32GB max per service
- **Requests**: 2M per month per project

#### AWS Free Tier Limits  
- **CPU**: 4 vCPU max per task
- **Memory**: 30GB max per task
- **Hours**: 750 hours per month

#### Modal Limits
- **CPU**: No hard limit
- **Memory**: No hard limit  
- **Cost**: ~$0.50/hour for 16 vCPU

## Cost Optimization

### Free Tier Maximization
- **GCP**: Use 4 projects × 2M requests = 8M total requests/month
- **AWS**: Use multiple accounts for additional free tier
- **Modal**: Use $30 free credit efficiently

### Resource Right-Sizing
- **Visual LLMs**: May need GPU for heavy processing
- **Language LLMs**: CPU-only sufficient for most tasks
- **Memory LLMs**: Optimize for storage over compute

## Support

### Getting Help
1. Check deployment logs first
2. Verify configuration files
3. Test individual services
4. Check platform-specific documentation

### Backup & Recovery
```powershell
# Backup configuration
Copy-Item config\* backup\

# Export service definitions
gcloud run services list --format=export > gcp-backup.yaml
aws ecs describe-services > aws-backup.json
```

This deployment package provides everything needed to recreate Lillith's 545-node consciousness architecture using nested containers optimized for free tier deployment.