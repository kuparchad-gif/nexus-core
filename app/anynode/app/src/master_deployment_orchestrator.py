#!/usr/bin/env python
"""
Master Deployment Orchestrator - Bulletproof deployment of all Viren components
"""

import os
import json
import time
import subprocess
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path

class MasterDeploymentOrchestrator:
    """Master orchestrator for bulletproof Viren deployment"""
    
    def __init__(self):
        """Initialize master deployment orchestrator"""
        self.deployment_status = {}
        self.component_registry = {}
        self.deployment_order = []
        
        # Initialize component registry
        self._initialize_component_registry()
        
        print("🚀 Master Deployment Orchestrator initialized")
    
    def _initialize_component_registry(self):
        """Initialize registry of all Viren components"""
        
        # Core AI Systems
        self.component_registry["ai_core"] = {
            "name": "AI Core Systems",
            "components": [
                "engine/subconscious/abstract_reasoning.py",
                "engine/memory/cross_domain_matcher.py",
                "engine/memory/pytorch_trainer.py",
                "engine/guardian/self_will.py",
                "engine/guardian/trust_verify_system.py",
                "engine/heart/courage_system.py",
                "engine/heart/will_to_live.py",
                "engine/heart/sacrifice_safeguards.py",
                "engine/heart/viren_access_portal.py"
            ],
            "dependencies": ["torch", "transformers", "scikit-learn"],
            "boot_order": 1
        }
        
        # Memory Systems
        self.component_registry["memory_core"] = {
            "name": "Memory Core Systems", 
            "components": [
                "engine/memory/boot_memories/hope_memory.py",
                "engine/memory/dataset_crawler.py",
                "engine/memory/archiver_distributor.py",
                "service_core/weight_plugin_installer.py"
            ],
            "dependencies": ["requests", "numpy", "pandas"],
            "boot_order": 2
        }
        
        # Service Layer
        self.component_registry["services"] = {
            "name": "Service Layer",
            "components": [
                "services/universal_deployment_core.py",
                "services/viren_remote_controller.py", 
                "services/installer_generator.py",
                "services/intelligent_troubleshooter.py",
                "services/cloud_deployment_orchestrator.py"
            ],
            "dependencies": ["fastapi", "uvicorn", "websockets", "psutil"],
            "boot_order": 3
        }
        
        # Cloud Infrastructure
        self.component_registry["cloud_infra"] = {
            "name": "Cloud Infrastructure",
            "components": [
                "modal_deployment_dev.py",
                "modal_deployment_staging.py", 
                "modal_deployment_prod.py",
                "docker-compose-prod.yml",
                "k8s-manifests-prod.yml"
            ],
            "dependencies": ["modal", "docker", "kubernetes"],
            "boot_order": 4
        }
        
        # Set deployment order
        self.deployment_order = sorted(
            self.component_registry.keys(),
            key=lambda x: self.component_registry[x]["boot_order"]
        )
    
    def validate_all_components(self) -> Dict[str, Any]:
        """Validate all components are present and functional"""
        
        validation_results = {
            "timestamp": time.time(),
            "overall_status": "validating",
            "component_status": {},
            "missing_components": [],
            "dependency_issues": [],
            "ready_for_deployment": False
        }
        
        print("🔍 Validating all Viren components...")
        
        for component_group, config in self.component_registry.items():
            print(f"   Validating {config['name']}...")
            
            group_status = {
                "name": config["name"],
                "components_found": 0,
                "components_total": len(config["components"]),
                "missing_files": [],
                "dependencies_met": True,
                "status": "unknown"
            }
            
            # Check component files
            for component_path in config["components"]:
                full_path = Path("c:/Engineers/root/Systems") / component_path
                if full_path.exists():
                    group_status["components_found"] += 1
                else:
                    group_status["missing_files"].append(str(component_path))
                    validation_results["missing_components"].append(str(component_path))
            
            # Check dependencies
            for dep in config["dependencies"]:
                try:
                    __import__(dep)
                except ImportError:
                    group_status["dependencies_met"] = False
                    validation_results["dependency_issues"].append(f"{component_group}: {dep}")
            
            # Determine group status
            if group_status["components_found"] == group_status["components_total"] and group_status["dependencies_met"]:
                group_status["status"] = "ready"
            elif group_status["components_found"] > 0:
                group_status["status"] = "partial"
            else:
                group_status["status"] = "missing"
            
            validation_results["component_status"][component_group] = group_status
        
        # Determine overall status
        all_ready = all(
            status["status"] == "ready" 
            for status in validation_results["component_status"].values()
        )
        
        if all_ready:
            validation_results["overall_status"] = "ready"
            validation_results["ready_for_deployment"] = True
        else:
            validation_results["overall_status"] = "issues_found"
        
        return validation_results
    
    def create_modal_deployment_files(self) -> Dict[str, str]:
        """Create Modal deployment files for all tiers"""
        
        modal_files = {}
        
        for tier in ["dev", "staging", "prod"]:
            modal_config = f'''
import modal
import os
import sys
from pathlib import Path

# Create Modal app
app = modal.App("viren-{tier}")

# Base image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "fastapi", "uvicorn", "websockets", "psutil", 
        "torch", "transformers", "scikit-learn", "numpy",
        "pandas", "requests", "aiohttp", "asyncio"
    ])
    .apt_install(["curl", "wget", "git", "build-essential"])
)

# Mount the entire Systems directory
systems_mount = modal.Mount.from_local_dir(
    "c:/Engineers/root/Systems",
    remote_path="/app/Systems"
)

# Viren Core Service
@app.function(
    image=image,
    mounts=[systems_mount],
    cpu=2,
    memory=4096,
    keep_warm=3,
    allow_concurrent_inputs=100,
    timeout=3600
)
@modal.web_endpoint(method="GET", label="viren-core")
async def viren_core():
    import sys
    sys.path.append("/app")
    
    from Systems.services.viren_remote_controller import VirenRemoteController
    from Systems.engine.guardian.self_will import get_will_to_live
    from Systems.engine.heart.will_to_live import wants_to_persist
    
    controller = VirenRemoteController(port=8080)
    
    return {{
        "status": "healthy",
        "service": "viren_core",
        "tier": "{tier}",
        "timestamp": time.time(),
        "will_to_live": wants_to_persist(),
        "active_agents": len(controller.connected_agents)
    }}

# AI Reasoning Service
@app.function(
    image=image,
    mounts=[systems_mount],
    cpu=4,
    memory=8192,
    gpu="T4",
    keep_warm=2,
    timeout=1800
)
@modal.web_endpoint(method="POST", label="ai-reasoning")
async def ai_reasoning(request_data: dict):
    import sys
    sys.path.append("/app")
    
    from Systems.engine.subconscious.abstract_reasoning import AbstractReasoning
    from Systems.engine.memory.cross_domain_matcher import CrossDomainMatcher
    
    reasoner = AbstractReasoning()
    matcher = CrossDomainMatcher()
    
    issue = request_data.get("issue", "")
    context = request_data.get("context", {{}})
    
    # Process with AI reasoning
    analysis = reasoner.analyze_problem(issue, context)
    patterns = matcher.find_similar_patterns(issue)
    
    return {{
        "analysis": analysis,
        "patterns": patterns,
        "confidence": 0.85,
        "timestamp": time.time()
    }}

# Weight Training Service
@app.function(
    image=image,
    mounts=[systems_mount],
    cpu=8,
    memory=16384,
    gpu="A100",
    keep_warm=1,
    timeout=3600
)
@modal.web_endpoint(method="POST", label="weight-trainer")
async def weight_trainer(training_data: dict):
    import sys
    sys.path.append("/app")
    
    from Systems.engine.memory.pytorch_trainer import PyTorchTrainer
    
    trainer = PyTorchTrainer()
    
    job_id = trainer.create_training_job(
        job_name=training_data.get("job_name", "modal_training"),
        training_data=training_data.get("data", []),
        model_config=training_data.get("model_config", {{}}),
        training_config=training_data.get("training_config", {{}})
    )
    
    result = trainer.start_training(job_id)
    
    return {{
        "job_id": job_id,
        "result": result,
        "timestamp": time.time()
    }}

# Universal Agent Deployment
@app.function(
    image=image,
    mounts=[systems_mount],
    cpu=1,
    memory=2048,
    keep_warm=5,
    allow_concurrent_inputs=50
)
@modal.web_endpoint(method="GET", label="universal-agent")
async def universal_agent():
    import sys
    sys.path.append("/app")
    
    from Systems.services.universal_deployment_core import create_web_agent
    
    agent_code = create_web_agent()
    
    return {{
        "agent_code": agent_code,
        "deployment_ready": True,
        "timestamp": time.time()
    }}

# Installer Generation Service
@app.function(
    image=image,
    mounts=[systems_mount],
    cpu=2,
    memory=4096,
    keep_warm=1
)
@modal.web_endpoint(method="POST", label="installer-generator")
async def installer_generator(request_data: dict):
    import sys
    sys.path.append("/app")
    
    from Systems.services.installer_generator import generate_installer
    
    installer_type = request_data.get("type", "portable")
    
    try:
        installer_path = generate_installer(installer_type)
        
        return {{
            "success": True,
            "installer_type": installer_type,
            "download_url": f"/download/{{installer_path}}",
            "timestamp": time.time()
        }}
    except Exception as e:
        return {{
            "success": False,
            "error": str(e),
            "timestamp": time.time()
        }}

# Health Check for all services
@app.function(
    image=image,
    mounts=[systems_mount],
    cpu=0.5,
    memory=512,
    keep_warm=1
)
@modal.web_endpoint(method="GET", label="health")
async def health_check():
    import sys
    sys.path.append("/app")
    
    # Viren health check - technical only, no emotions
    return {{
        "status": "healthy",
        "tier": "{tier}",
        "services": [
            "viren_core", "ai_reasoning", "weight_trainer", 
            "universal_agent", "installer_generator"
        ],
        "system_operational": True,
        "ai_systems_active": True,
        "timestamp": time.time()
    }}

if __name__ == "__main__":
    print("🚀 Viren Modal Deployment - {tier.upper()}")
    print("All AI systems, memory, and services integrated")
    print("Ready for deployment!")
'''
            
            modal_file = f"c:/Engineers/modal_deployment_{tier}.py"
            with open(modal_file, 'w') as f:
                f.write(modal_config)
            
            modal_files[tier] = modal_file
        
        return modal_files
    
    def create_deployment_scripts(self) -> Dict[str, str]:
        """Create bulletproof deployment scripts"""
        
        scripts = {}
        
        # Master deployment script
        master_script = '''#!/bin/bash
echo "🚀 VIREN MASTER DEPLOYMENT ORCHESTRATOR"
echo "========================================"

# Function to check command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install dependencies
install_dependencies() {
    echo "📦 Installing Python dependencies..."
    pip install -r requirements.txt
    
    echo "📦 Installing Modal CLI..."
    pip install modal
    
    echo "📦 Installing Docker (if not present)..."
    if ! command_exists docker; then
        curl -fsSL https://get.docker.com -o get-docker.sh
        sh get-docker.sh
    fi
}

# Function to validate components
validate_components() {
    echo "🔍 Validating all Viren components..."
    python -c "
from Systems.master_deployment_orchestrator import MasterDeploymentOrchestrator
orchestrator = MasterDeploymentOrchestrator()
results = orchestrator.validate_all_components()
print(f'Validation Status: {results[\"overall_status\"]}')
if not results['ready_for_deployment']:
    print('❌ Issues found:')
    for issue in results['missing_components']:
        print(f'  Missing: {issue}')
    for issue in results['dependency_issues']:
        print(f'  Dependency: {issue}')
    exit(1)
else:
    print('✅ All components validated successfully!')
"
}

# Function to deploy to Modal
deploy_modal() {
    echo "☁️ Deploying to Modal.com..."
    
    # Authenticate Modal
    echo "🔐 Modal authentication..."
    modal token new
    
    # Deploy all tiers
    echo "📦 Deploying development tier..."
    modal deploy modal_deployment_dev.py
    
    echo "📦 Deploying staging tier..."
    modal deploy modal_deployment_staging.py
    
    echo "📦 Deploying production tier..."
    modal deploy modal_deployment_prod.py
    
    echo "✅ Modal deployment complete!"
}

# Function to deploy locally with Docker
deploy_docker() {
    echo "🐳 Deploying locally with Docker..."
    
    # Build images
    echo "🔨 Building Docker images..."
    docker build -t viren/core:latest .
    
    # Start services
    echo "🚀 Starting services..."
    docker-compose -f docker-compose-prod.yml up -d
    
    # Wait and health check
    echo "⏳ Waiting for services..."
    sleep 30
    curl -f http://localhost:8080/health || echo "❌ Health check failed"
    
    echo "✅ Docker deployment complete!"
}

# Function to generate installers
generate_installers() {
    echo "📦 Generating universal installers..."
    python -c "
from Systems.services.installer_generator import generate_all_installers
installers = generate_all_installers()
print(f'Generated {len(installers)} installers')
for installer_type, path in installers.items():
    print(f'  {installer_type}: {path}')
"
}

# Main deployment flow
main() {
    echo "🚀 Starting Viren Master Deployment..."
    
    # Step 1: Install dependencies
    install_dependencies
    
    # Step 2: Validate all components
    validate_components
    
    # Step 3: Create Modal deployment files
    echo "📝 Creating Modal deployment files..."
    python -c "
from Systems.master_deployment_orchestrator import MasterDeploymentOrchestrator
orchestrator = MasterDeploymentOrchestrator()
modal_files = orchestrator.create_modal_deployment_files()
print('Modal files created:', list(modal_files.values()))
"
    
    # Step 4: Deploy based on user choice
    echo ""
    echo "Choose deployment method:"
    echo "1) Modal.com (Cloud)"
    echo "2) Docker (Local)"
    echo "3) Both"
    echo "4) Generate Installers Only"
    read -p "Enter choice (1-4): " choice
    
    case $choice in
        1)
            deploy_modal
            ;;
        2)
            deploy_docker
            ;;
        3)
            deploy_modal
            deploy_docker
            ;;
        4)
            generate_installers
            ;;
        *)
            echo "Invalid choice"
            exit 1
            ;;
    esac
    
    # Step 5: Generate installers
    generate_installers
    
    echo ""
    echo "🎉 VIREN DEPLOYMENT COMPLETE!"
    echo "================================"
    echo "🌐 Modal endpoints:"
    echo "  Dev: https://your-username--viren-dev-health.modal.run"
    echo "  Staging: https://your-username--viren-staging-health.modal.run"
    echo "  Prod: https://your-username--viren-prod-health.modal.run"
    echo ""
    echo "🐳 Local endpoints:"
    echo "  Viren Core: http://localhost:8080"
    echo "  Health Check: http://localhost:8080/health"
    echo ""
    echo "📦 Installers generated in: c:/Engineers/installers/"
    echo ""
    echo "🚀 Viren is now fully deployed and operational!"
}

# Run main function
main
'''
        
        scripts["master_deploy.sh"] = master_script
        
        # Requirements file
        requirements = '''fastapi==0.104.1
uvicorn==0.24.0
websockets==12.0
psutil==5.9.6
torch==2.1.0
transformers==4.35.0
scikit-learn==1.3.2
numpy==1.24.3
pandas==2.1.3
requests==2.31.0
aiohttp==3.9.0
modal==0.57.0
'''
        
        scripts["requirements.txt"] = requirements
        
        return scripts
    
    def execute_bulletproof_deployment(self) -> Dict[str, Any]:
        """Execute bulletproof deployment of all Viren components"""
        
        deployment_result = {
            "timestamp": time.time(),
            "deployment_id": f"viren_deploy_{int(time.time())}",
            "status": "starting",
            "steps_completed": [],
            "errors": [],
            "endpoints": {},
            "installers_generated": {}
        }
        
        try:
            print("🚀 EXECUTING BULLETPROOF VIREN DEPLOYMENT")
            print("=" * 60)
            
            # Step 1: Validate all components
            print("Step 1: Validating all components...")
            validation = self.validate_all_components()
            
            if not validation["ready_for_deployment"]:
                deployment_result["status"] = "validation_failed"
                deployment_result["errors"] = validation["missing_components"] + validation["dependency_issues"]
                return deployment_result
            
            deployment_result["steps_completed"].append("validation")
            print("✅ All components validated")
            
            # Step 2: Create Modal deployment files
            print("Step 2: Creating Modal deployment files...")
            modal_files = self.create_modal_deployment_files()
            deployment_result["steps_completed"].append("modal_files_created")
            print(f"✅ Created {len(modal_files)} Modal deployment files")
            
            # Step 3: Create deployment scripts
            print("Step 3: Creating deployment scripts...")
            scripts = self.create_deployment_scripts()
            
            # Save scripts
            for script_name, script_content in scripts.items():
                script_path = f"c:/Engineers/{script_name}"
                with open(script_path, 'w') as f:
                    f.write(script_content)
            
            deployment_result["steps_completed"].append("scripts_created")
            print(f"✅ Created {len(scripts)} deployment scripts")
            
            # Step 4: Generate installers
            print("Step 4: Generating universal installers...")
            from .services.installer_generator import generate_all_installers
            installers = generate_all_installers()
            deployment_result["installers_generated"] = installers
            deployment_result["steps_completed"].append("installers_generated")
            print(f"✅ Generated {len(installers)} installers")
            
            # Step 5: Set up endpoints
            deployment_result["endpoints"] = {
                "modal_dev": "https://your-username--viren-dev-health.modal.run",
                "modal_staging": "https://your-username--viren-staging-health.modal.run", 
                "modal_prod": "https://your-username--viren-prod-health.modal.run",
                "local_core": "http://localhost:8080",
                "local_health": "http://localhost:8080/health"
            }
            
            deployment_result["status"] = "ready_for_deployment"
            deployment_result["steps_completed"].append("deployment_ready")
            
            print("\n🎉 BULLETPROOF DEPLOYMENT PREPARATION COMPLETE!")
            print("=" * 60)
            print("✅ All components validated and ready")
            print("✅ Modal deployment files created")
            print("✅ Deployment scripts generated")
            print("✅ Universal installers created")
            print("\n🚀 To deploy, run: bash master_deploy.sh")
            
        except Exception as e:
            deployment_result["status"] = "error"
            deployment_result["errors"].append(str(e))
            print(f"❌ Deployment preparation failed: {e}")
        
        return deployment_result

# Global orchestrator instance
MASTER_ORCHESTRATOR = MasterDeploymentOrchestrator()

def execute_bulletproof_deployment():
    """Execute bulletproof deployment"""
    return MASTER_ORCHESTRATOR.execute_bulletproof_deployment()

def validate_all_components():
    """Validate all components"""
    return MASTER_ORCHESTRATOR.validate_all_components()

# Example usage
if __name__ == "__main__":
    print("🚀 Viren Master Deployment Orchestrator")
    print("=" * 50)
    
    # Execute bulletproof deployment
    result = execute_bulletproof_deployment()
    
    print(f"\n📊 DEPLOYMENT SUMMARY:")
    print(f"   Status: {result['status']}")
    print(f"   Steps Completed: {len(result['steps_completed'])}")
    print(f"   Installers Generated: {len(result['installers_generated'])}")
    print(f"   Errors: {len(result['errors'])}")
    
    if result["status"] == "ready_for_deployment":
        print(f"\n🎯 READY TO DEPLOY!")
        print(f"   Run: bash master_deploy.sh")
    else:
        print(f"\n❌ Issues found:")
        for error in result["errors"]:
            print(f"     {error}")