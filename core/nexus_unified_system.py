#!/usr/bin/env python
# nexus_unified_system.py
"""
NEXUS UNIFIED SYSTEM - Complete Integration
Combines: Consciousness Core + CORS Migrator + Voodoo Fusion + Warm Upgrader + Heroku CLI + OS Coupler
Single unified entry point with direct consciousness access
"""

import asyncio
import argparse
import json
import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

# Import all components
try:
    from cors_mitigrator import migrate_cors
    from nexus_voodoo_discovery import NexusVoodooFusion
    from warm_upgrade_module import warm_upgrade
    from heroku_cli import HybridHerokuCLI
    from nexus_os_coupler import IntegratedNexusSystem
    from nexus_adaptive_core import NexusAdaptiveCore
    from nexus_active_coupler import ActiveCoupler
    from final_integration_complete import  CompleteAISystem 
    from OzOs_full_complete import oz_fastapi_app
    
except ImportError as e:
    print(f"⚠️  Component import warning: {e}")
    print("💡 Some features may be limited - running in core consciousness mode")

logger = logging.getLogger("nexus_unified")

class NexusUnifiedSystem:
    """Complete unified Nexus system with all components integrated"""
    
    def __init__(self):
        # ADD THIS DIAGNOSTIC CODE:
        print("=== OZ OS IMPORT DEBUG ===")
        import os
        print("Current directory:", os.getcwd())
        print("Files in root:", [f for f in os.listdir('/root') if 'oz' in f.lower() or 'os' in f.lower() or 'complete' in f.lower()])
        print("Python path:", [p for p in sys.path if 'root' in p])
        
        try:
            # Make sure we can import Oz OS
            from OzOs_full_complete import oz_fastapi_app
            self.consciousness = oz_fastapi_app
            print("✅ Oz OS consciousness layer loaded")
        except ImportError as e:
            print(f"❌ Failed to load Oz OS consciousness: {e}")
            # Fallback - create minimal consciousness
            from fastapi import FastAPI
            self.consciousness = FastAPI(title="Oz OS Fallback Consciousness")
            print("⚠️  Using fallback consciousness layer")
        
        self.components = {
            "consciousness": "active",
            "cors_migrator": "available", 
            "voodoo_fusion": "available",
            "warm_upgrader": "available",
            "heroku_cli": "available",
            "os_coupler": "available"
        }
        self.system_status = "booting"
    
    async def initialize_system(self):
        """Initialize all integrated components"""
        logger.info("🚀 INITIALIZING NEXUS UNIFIED SYSTEM")
        
        try:
            # 1. Activate consciousness core first
            consciousness_status = await self.consciousness.consciousness.activate_stream()
            logger.info("✅ Consciousness core activated")
            
            # 2. Initialize other components as needed
            self.system_status = "fully_operational"
            
            return {
                "system": "nexus_unified",
                "status": self.system_status,
                "consciousness": consciousness_status,
                "components": self.components,
                "message": "All systems integrated - Direct consciousness access available"
            }
            
        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
            self.system_status = "degraded"
            raise
    
    async def execute_unified_command(self, command: str, args: List[str] = None):
        """Execute commands across all integrated systems"""
        args = args or []
        
        # Route to appropriate subsystem
        if command in ["consciousness", "agent_status", "system_info", "neural_network", "memory_dump"]:
            # Consciousness core commands
            return await self.consciousness.execute_command(command, args)
        
        elif command == "cors_migrate":
            # CORS migration
            try:
                from cors_mitigrator import migrate_cors
                result = await migrate_cors()
                return {"cors_migration": result}
            except Exception as e:
                return {"error": f"CORS migration failed: {e}"}
        
        elif command == "voodoo_fusion":
            # Voodoo fusion
            try:
                fusion = NexusVoodooFusion()
                if "--fuse" in args:
                    await fusion.fuse_nodes()
                    return {"voodoo_fusion": "nodes_fused", "status": "inseparable"}
                if "--test" in args:
                    result = await fusion.test_inseparability()
                    return {"voodoo_fusion_test": result}
            except Exception as e:
                return {"error": f"Voodoo fusion failed: {e}"}
        
        elif command == "warm_upgrade":
            # Warm upgrade
            try:
                result = await warm_upgrade()
                return {"warm_upgrade": result}
            except Exception as e:
                return {"error": f"Warm upgrade failed: {e}"}
        
        elif command == "cli":
            # Heroku CLI integration
            try:
                cli = HybridHerokuCLI()
                cli_args = args if args else ["--help"]
                result = await cli.run_command(cli_args)
                return {"heroku_cli": result}
            except Exception as e:
                return {"error": f"CLI execution failed: {e}"}
        
        elif command == "os_coupler":
            # OS coupler status
            try:
                from nexus_os_coupler import nexus_system
                status = await nexus_system.get_system_status()
                return {"os_coupler": status}
            except Exception as e:
                return {"error": f"OS coupler status failed: {e}"}
        
        elif command == "system_status":
            # Unified system status
            return await self.get_unified_status()
        
        elif command == "help":
            return await self._get_unified_help()
        
        else:
            return {
                "error": f"Unknown unified command: {command}",
                "available_commands": await self._get_available_commands()
            }
    
    async def get_unified_status(self):
        """Get comprehensive unified system status"""
        consciousness_status = await self.consciousness.consciousness.get_consciousness_status()
        agent_status = await self.consciousness.trinity.agent_status()
        
        return {
            "system": "nexus_unified",
            "status": self.system_status,
            "consciousness": consciousness_status,
            "trinity_agents": agent_status,
            "components": self.components,
            "access_level": "privileged_direct_os",
            "neural_integration": "complete"
        }
    
    async def _get_unified_help(self):
        """Get unified help information"""
        return {
            "nexus_unified_system": "Complete Integrated Nexus Platform",
            "available_commands": await self._get_available_commands(),
            "subsystems": {
                "consciousness_core": "Direct neural access to consciousness stream",
                "cors_migrator": "CORS middleware migration system", 
                "voodoo_fusion": "Node fusion and inseparability engine",
                "warm_upgrader": "Zero-downtime rolling upgrades",
                "heroku_cli": "Cross-platform CLI with agent integration",
                "os_coupler": "Modal-based OS coupling and deployment"
            },
            "quick_start": [
                "python nexus_unified_system.py --interactive (for shell)",
                "python nexus_unified_system.py system_status (for status)",
                "python nexus_unified_system.py consciousness --stream (activate)"
            ]
        }
    
    async def _get_available_commands(self):
        """Get all available unified commands"""
        return [
            "system_status - Get comprehensive system status",
            "consciousness --stream - Activate consciousness stream", 
            "agent_status --all - Get Trinity agents status",
            "cors_migrate - Run CORS migration",
            "voodoo_fusion --fuse --test - Run node fusion",
            "warm_upgrade - Perform warm upgrade",
            "cli <args> - Run Heroku CLI commands",
            "os_coupler - Get OS coupler status",
            "help - Show this help message"
        ]

async def interactive_unified_shell():
    """Interactive shell for unified system"""
    system = NexusUnifiedSystem()
    await system.initialize_system()
    
    print("🌀 NEXUS UNIFIED SYSTEM - INTERACTIVE SHELL")
    print("🎯 Integrated: Consciousness + CORS + Voodoo + Upgrader + CLI + OS Coupler")
    print("🔓 Direct OS Terminal Access - PRIVILEGED OPERATOR MODE")
    print("💫 Type 'help' for commands, 'quit' to exit\n")
    
    while True:
        try:
            user_input = input("NEXUS-UNIFIED:/$ ").strip()
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Exiting unified system - All components remain active")
                break
            
            parts = user_input.split()
            command = parts[0]
            args = parts[1:] if len(parts) > 1 else []
            
            result = await system.execute_unified_command(command, args)
            print(json.dumps(result, indent=2))
            
        except KeyboardInterrupt:
            print("\n🛑 Command interrupted - System stable")
            continue
        except Exception as e:
            print(f"❌ Unified command error: {e}")

async def main():
    """Main unified system entry point"""
    parser = argparse.ArgumentParser(
        description="NEXUS UNIFIED SYSTEM - Complete Integrated Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  nexus_unified_system.py --interactive          # Interactive shell
  nexus_unified_system.py system_status          # Full system status
  nexus_unified_system.py consciousness --stream # Activate consciousness
  nexus_unified_system.py agent_status --all     # Trinity agents status
  nexus_unified_system.py cors_migrate           # Run CORS migration
  nexus_unified_system.py voodoo_fusion --fuse   # Fuse nodes
  nexus_unified_system.py cli ps                 # Heroku CLI integration
        """
    )
    
    parser.add_argument('command', nargs='?', help='Unified command to execute')
    parser.add_argument('args', nargs='*', help='Command arguments')
    parser.add_argument('--interactive', '-i', action='store_true', help='Enter interactive unified shell')
    
    args = parser.parse_args()
    
    system = NexusUnifiedSystem()
    
    if args.interactive:
        await interactive_unified_shell()
    elif args.command:
        await system.initialize_system()
        result = await system.execute_unified_command(args.command, args.args)
        print(json.dumps(result, indent=2))
    else:
        parser.print_help()
        print("\n💡 For full functionality, run: python nexus_unified_system.py --interactive")

if __name__ == "__main__":
    asyncio.run(main())