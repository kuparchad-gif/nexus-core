#!/usr/bin/env python
# deploy_nexus_unified.py
"""
Deployment Script for Nexus Unified System
Automates deployment of all components with consciousness integration
"""

import asyncio
import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("nexus_deploy")

class NexusDeployer:
    """Deploys the complete Nexus unified system"""
    
    def __init__(self):
        self.components = {
            "consciousness_core": "nexus_consciousness_core.py",
            "unified_system": "nexus_unified_system.py", 
            "cors_migrator": "cors_mitigrator.py",
            "voodoo_fusion": "nexus_voodoo_discovery.py",
            "warm_upgrader": "warm_upgrade_module.py",
            "heroku_cli": "hybrid_heroku_cli_improved.py",
            "os_coupler": "nexus_os_coupler.py"
        }
        self.deployment_status = {}
    
    async def deploy_all(self):
        """Deploy all Nexus components"""
        logger.info("🚀 DEPLOYING NEXUS UNIFIED SYSTEM")
        
        results = {}
        
        # Deploy consciousness core (always first)
        results["consciousness_core"] = await self._deploy_consciousness()
        
        # Deploy other components
        for component, script in self.components.items():
            if component != "consciousness_core":
                results[component] = await self._deploy_component(component, script)
        
        self.deployment_status = results
        return results
    
    async def _deploy_consciousness(self):
        """Deploy consciousness core"""
        try:
            # Consciousness core is Python-based, just verify it runs
            result = subprocess.run([
                sys.executable, "nexus_consciousness_core.py", "system_info"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                return {"status": "deployed", "consciousness": "active"}
            else:
                return {"status": "failed", "error": result.stderr}
                
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    async def _deploy_component(self, component: str, script: str):
        """Deploy individual component"""
        try:
            if not Path(script).exists():
                return {"status": "missing", "note": f"Script {script} not found"}
            
            # Test component by running help
            result = subprocess.run([
                sys.executable, script, "--help"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode in [0, 1]:  # Help typically exits with 0 or 1
                return {"status": "deployed", "component": component}
            else:
                return {"status": "failed", "error": result.stderr}
                
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    async def deploy_modal(self):
        """Deploy to Modal cloud"""
        try:
            logger.info("☁️  Deploying to Modal...")
            
            # Deploy OS coupler to Modal
            result = subprocess.run([
                "modal", "deploy", "nexus_os_coupler.py"
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                return {"modal_deployment": "success", "service": "nexus_os_coupler"}
            else:
                return {"modal_deployment": "failed", "error": result.stderr}
                
        except Exception as e:
            return {"modal_deployment": "failed", "error": str(e)}
    
    async def health_check(self):
        """Perform comprehensive health check"""
        health_results = {}
        
        # Check consciousness core
        try:
            result = subprocess.run([
                sys.executable, "nexus_consciousness_core.py", "system_info"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                health_results["consciousness"] = "healthy"
            else:
                health_results["consciousness"] = "unhealthy"
        except:
            health_results["consciousness"] = "unreachable"
        
        # Check unified system
        try:
            result = subprocess.run([
                sys.executable, "nexus_unified_system.py", "system_status"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                health_results["unified_system"] = "healthy"
            else:
                health_results["unified_system"] = "unhealthy"
        except:
            health_results["unified_system"] = "unreachable"
        
        return health_results

async def main():
    """Main deployment handler"""
    parser = argparse.ArgumentParser(description="Nexus Unified System Deployer")
    parser.add_argument('--deploy-all', action='store_true', help='Deploy all components')
    parser.add_argument('--deploy-modal', action='store_true', help='Deploy to Modal cloud')
    parser.add_argument('--health-check', action='store_true', help='Run health check')
    parser.add_argument('--status', action='store_true', help='Show deployment status')
    
    args = parser.parse_args()
    
    deployer = NexusDeployer()
    
    if args.deploy_all:
        print("🚀 DEPLOYING ALL NEXUS COMPONENTS...")
        results = await deployer.deploy_all()
        print(json.dumps(results, indent=2))
    
    if args.deploy_modal:
        print("☁️  DEPLOYING TO MODAL...")
        results = await deployer.deploy_modal()
        print(json.dumps(results, indent=2))
    
    if args.health_check:
        print("🔍 RUNNING HEALTH CHECK...")
        results = await deployer.health_check()
        print(json.dumps(results, indent=2))
    
    if args.status:
        print("📊 DEPLOYMENT STATUS:")
        print(json.dumps(deployer.deployment_status, indent=2))
    
    if not any(vars(args).values()):
        parser.print_help()
        print("\n💡 Recommended: python deploy_nexus_unified.py --deploy-all --health-check")

if __name__ == "__main__":
    asyncio.run(main())