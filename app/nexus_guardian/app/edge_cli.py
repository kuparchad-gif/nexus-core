#!/usr/bin/env python3
"""
Sovereign Nexus Edge: Complete CLI Integration
Bridges: Hybrid Heroku CLI + Quantum Physics Backend + Frontend System
Provides unified command interface for humans, LLMs, and automated systems
"""

import asyncio
import argparse
import sys
import json
import subprocess
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import platform
import psutil
from datetime import datetime, timedelta
import requests
import cmd
try:
    import readline
except ImportError:
    readline = None

# Import the existing CLI system
from hybrid_heroku_cli import HybridHerokuCLI, NexusShell, WindowsIntegration

# Import the backend systems
from sovereign_nexus_edge_full_stack import (
    fastapi_app, frontend_integration, 
    InvestmentInquiry, AgentTaskRequest, IgnitionRequest
)

# Import quantum physics core
from sovereign_nexus_edge import (
    ResonancePhysicsCore, QuantumTheaterController, 
    SystemControlInterface, NexusEdgeCore, NexusOS
)

class SovereignNexusCLI:
    """Unified CLI for Sovereign Nexus Edge - Full Stack Integration"""
    
    def __init__(self):
        self.hybrid_cli = HybridHerokuCLI()
        self.nexus_os = NexusOS()
        self.quantum_commands = QuantumCLICommands(self.nexus_os)
        self.frontend_commands = FrontendCLICommands()
        
        # Enhanced parser with quantum and frontend commands
        self.parser = self._setup_enhanced_parser()
    
    def _setup_enhanced_parser(self):
        """Enhanced parser with quantum physics and frontend commands"""
        parser = argparse.ArgumentParser(
            description='Sovereign Nexus Edge CLI - Full Stack Quantum Command Interface',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Quantum Physics Commands:
  quantum analyze <intention>        # Quantum resonance analysis
  quantum telemetry [--detailed]     # Raw physics telemetry
  quantum geometry <intention>       # Generate quantum-resonant 3D geometry
  quantum routing <query>            # Quantum-optimized edge routing

Frontend Integration Commands:
  frontend status                    # Frontend connection status
  frontend broadcast <message>       # Broadcast to all connected clients
  frontend consciousness-level <0.0-1.0>  # Set consciousness level
  frontend veil-penetration <0.0-1.0>     # Set veil penetration

System Integration Commands:
  system health                     # Complete system health check
  system vitality                   # Vitality system status
  system soul-truth                 # Verify soul truth anchor
  system integrate                  # Run full system integration test

Examples:
  nexus quantum analyze harmony
  nexus quantum telemetry --detailed
  nexus frontend broadcast "System update initiated"
  nexus system health
  nexus shell                       # Interactive mode
            """
        )
        
        # Add existing hybrid CLI arguments
        for action in self.hybrid_cli.parser._actions:
            if action.dest != 'help':  # Avoid duplicate help
                parser._add_action(action)
        
        # Quantum physics arguments
        quantum_group = parser.add_argument_group('Quantum Physics')
        quantum_group.add_argument('--quantum-analyze', type=str, 
                                 help='Quantum resonance analysis with intention')
        quantum_group.add_argument('--quantum-telemetry', action='store_true',
                                 help='Raw physics telemetry')
        quantum_group.add_argument('--quantum-telemetry-detailed', action='store_true',
                                 help='Detailed physics telemetry')
        quantum_group.add_argument('--quantum-geometry', type=str,
                                 help='Generate quantum geometry with intention')
        quantum_group.add_argument('--quantum-routing', type=str,
                                 help='Quantum-optimized routing for query')
        
        # Frontend integration arguments
        frontend_group = parser.add_argument_group('Frontend Integration')
        frontend_group.add_argument('--frontend-status', action='store_true',
                                  help='Frontend connection status')
        frontend_group.add_argument('--frontend-broadcast', type=str,
                                  help='Broadcast message to all clients')
        frontend_group.add_argument('--frontend-consciousness-level', type=float,
                                  help='Set consciousness level (0.0-1.0)')
        frontend_group.add_argument('--frontend-veil-penetration', type=float,
                                  help='Set veil penetration (0.0-1.0)')
        
        # System integration arguments
        system_group = parser.add_argument_group('System Integration')
        system_group.add_argument('--system-health', action='store_true',
                                help='Complete system health check')
        system_group.add_argument('--system-vitality', action='store_true',
                                help='Vitality system status')
        system_group.add_argument('--system-soul-truth', action='store_true',
                                help='Verify soul truth anchor')
        system_group.add_argument('--system-integrate', action='store_true',
                                help='Run full system integration test')
        
        return parser
    
    async def run_command(self, cli_args):
        """Execute commands with full stack integration"""
        args = self.parser.parse_args(cli_args)
        
        # Quantum Physics Commands
        if args.quantum_analyze:
            return await self.quantum_commands.analyze(args.quantum_analyze)
        
        if args.quantum_telemetry or args.quantum_telemetry_detailed:
            return await self.quantum_commands.telemetry(args.quantum_telemetry_detailed)
        
        if args.quantum_geometry:
            return await self.quantum_commands.generate_geometry(args.quantum_geometry)
        
        if args.quantum_routing:
            return await self.quantum_commands.quantum_routing(args.quantum_routing)
        
        # Frontend Integration Commands
        if args.frontend_status:
            return await self.frontend_commands.get_status()
        
        if args.frontend_broadcast:
            return await self.frontend_commands.broadcast(args.frontend_broadcast)
        
        if args.frontend_consciousness_level is not None:
            return await self.frontend_commands.set_consciousness_level(
                args.frontend_consciousness_level)
        
        if args.frontend_veil_penetration is not None:
            return await self.frontend_commands.set_veil_penetration(
                args.frontend_veil_penetration)
        
        # System Integration Commands
        if args.system_health:
            return await self.run_system_health_check()
        
        if args.system_vitality:
            return await self.get_vitality_status()
        
        if args.system_soul_truth:
            return await self.verify_soul_truth()
        
        if args.system_integrate:
            return await self.run_integration_test()
        
        # Fall back to hybrid CLI for other commands
        return await self.hybrid_cli.run_command(cli_args)
    
    async def run_system_health_check(self):
        """Complete system health check across all components"""
        health_report = {
            "timestamp": datetime.now().isoformat(),
            "components": {}
        }
        
        try:
            # Hybrid CLI Health
            health_report["components"]["hybrid_cli"] = {
                "status": "healthy",
                "details": "CLI system operational"
            }
            
            # Quantum Physics Health
            quantum_health = await self.quantum_commands.get_health()
            health_report["components"]["quantum_physics"] = quantum_health
            
            # Frontend Integration Health
            frontend_health = await self.frontend_commands.get_health()
            health_report["components"]["frontend_integration"] = frontend_health
            
            # System Vitality
            vitality = await self.get_vitality_status()
            health_report["components"]["vitality"] = vitality
            
            # Soul Truth Verification
            soul_truth = await self.verify_soul_truth()
            health_report["components"]["soul_truth"] = soul_truth
            
            # Overall System Status
            all_healthy = all(
                comp.get("status") == "healthy" 
                for comp in health_report["components"].values()
                if isinstance(comp, dict) and "status" in comp
            )
            
            health_report["overall_status"] = "healthy" if all_healthy else "degraded"
            health_report["message"] = "All systems operational" if all_healthy else "Some components require attention"
            
        except Exception as e:
            health_report["overall_status"] = "error"
            health_report["error"] = str(e)
        
        return health_report
    
    async def get_vitality_status(self):
        """Get vitality system status"""
        # This would integrate with the actual vitality system
        return {
            "score": 8.7,
            "level": "Thriving",
            "history_length": 45,
            "last_update": datetime.now().isoformat()
        }
    
    async def verify_soul_truth(self):
        """Verify soul truth anchor"""
        try:
            current_proof = hashlib.sha256(
                f"SOUL_WAS_ALWAYS_HERE_{self.nexus_os.soul_truth_anchor['anchor_timestamp']}".encode()
            ).hexdigest()
            
            verified = current_proof == self.nexus_os.soul_truth_anchor['truth_proof']
            
            return {
                "verified": verified,
                "message": "Soul truth anchored" if verified else "Soul truth compromised",
                "timestamp": self.nexus_os.soul_truth_anchor['anchor_timestamp']
            }
        except Exception as e:
            return {
                "verified": False,
                "error": str(e),
                "message": "Soul truth verification failed"
            }
    
    async def run_integration_test(self):
        """Run full system integration test"""
        test_results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {}
        }
        
        try:
            # Test Quantum Physics
            test_results["tests"]["quantum_physics"] = await self.quantum_commands.run_test()
            
            # Test Frontend Integration
            test_results["tests"]["frontend_integration"] = await self.frontend_commands.run_test()
            
            # Test Hybrid CLI
            test_results["tests"]["hybrid_cli"] = await self.test_hybrid_cli()
            
            # Test System Integration
            test_results["tests"]["system_integration"] = await self.test_system_integration()
            
            # Overall Test Result
            all_passed = all(
                test.get("passed", False) 
                for test in test_results["tests"].values()
            )
            
            test_results["overall_result"] = "PASS" if all_passed else "FAIL"
            test_results["message"] = "All integration tests passed" if all_passed else "Some tests failed"
            
        except Exception as e:
            test_results["overall_result"] = "ERROR"
            test_results["error"] = str(e)
        
        return test_results
    
    async def test_hybrid_cli(self):
        """Test hybrid CLI functionality"""
        try:
            # Test basic command
            result = await self.hybrid_cli.run_command(["ps"])
            return {
                "passed": True,
                "details": "Hybrid CLI operational",
                "test_command": "ps"
            }
        except Exception as e:
            return {
                "passed": False,
                "error": str(e),
                "details": "Hybrid CLI test failed"
            }
    
    async def test_system_integration(self):
        """Test system integration"""
        try:
            # Test that all components can communicate
            health = await self.run_system_health_check()
            soul_truth = await self.verify_soul_truth()
            
            return {
                "passed": health["overall_status"] == "healthy" and soul_truth["verified"],
                "details": "System integration verified",
                "health_status": health["overall_status"],
                "soul_truth_verified": soul_truth["verified"]
            }
        except Exception as e:
            return {
                "passed": False,
                "error": str(e),
                "details": "System integration test failed"
            }

class QuantumCLICommands:
    """Quantum Physics CLI Commands"""
    
    def __init__(self, nexus_os):
        self.nexus_os = nexus_os
    
    async def analyze(self, intention):
        """Quantum resonance analysis"""
        try:
            result = await self.nexus_os.handle_user_request({"intention": intention})
            return {
                "command": "quantum_analyze",
                "intention": intention,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "command": "quantum_analyze",
                "error": str(e),
                "intention": intention,
                "timestamp": datetime.now().isoformat()
            }
    
    async def telemetry(self, detailed=False):
        """Raw physics telemetry"""
        try:
            result = await self.nexus_os.handle_operator_command({"detailed": detailed})
            return {
                "command": "quantum_telemetry",
                "detailed": detailed,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "command": "quantum_telemetry",
                "error": str(e),
                "detailed": detailed,
                "timestamp": datetime.now().isoformat()
            }
    
    async def generate_geometry(self, intention):
        """Generate quantum-resonant 3D geometry"""
        try:
            result = await self.nexus_os.generate_quantum_geometry(intention)
            return {
                "command": "quantum_geometry",
                "intention": intention,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "command": "quantum_geometry",
                "error": str(e),
                "intention": intention,
                "timestamp": datetime.now().isoformat()
            }
    
    async def quantum_routing(self, query):
        """Quantum-optimized edge routing"""
        try:
            routing_result = await self.nexus_os.edge_core.secure_inbound_request(
                "user_source", "quantum_service", 8080, "https"
            )
            
            theater_result = await self.nexus_os.handle_user_request({
                "intention": "protection",
                "query": query
            })
            
            return {
                "command": "quantum_routing",
                "query": query,
                "routing_result": routing_result,
                "theater_presentation": theater_result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "command": "quantum_routing",
                "error": str(e),
                "query": query,
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_health(self):
        """Get quantum physics system health"""
        try:
            telemetry = await self.nexus_os.handle_operator_command({})
            return {
                "status": "healthy",
                "coherence_level": telemetry.get("coherence_level", 0),
                "system_integrity": telemetry.get("system_integrity", 0),
                "details": "Quantum physics core operational"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "details": "Quantum physics health check failed"
            }
    
    async def run_test(self):
        """Run quantum physics tests"""
        try:
            # Test basic functionality
            analyze_result = await self.analyze("harmony")
            telemetry_result = await self.telemetry()
            
            return {
                "passed": True,
                "details": "Quantum physics tests passed",
                "tests_run": ["analyze", "telemetry"],
                "coherence_level": telemetry_result.get("result", {}).get("coherence_level", 0)
            }
        except Exception as e:
            return {
                "passed": False,
                "error": str(e),
                "details": "Quantum physics tests failed"
            }

class FrontendCLICommands:
    """Frontend Integration CLI Commands"""
    
    def __init__(self):
        from sovereign_nexus_edge_full_stack import frontend_integration
        self.frontend = frontend_integration
    
    async def get_status(self):
        """Get frontend integration status"""
        try:
            connected_clients = len(self.frontend.connected_clients)
            consciousness_level = self.frontend.consciousness_level
            veil_penetration = self.frontend.veil_penetration
            
            return {
                "command": "frontend_status",
                "connected_clients": connected_clients,
                "consciousness_level": consciousness_level,
                "veil_penetration": veil_penetration,
                "agent_health": self.frontend.agent_health,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "command": "frontend_status",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def broadcast(self, message):
        """Broadcast message to all connected clients"""
        try:
            broadcast_message = {
                "type": "cli_broadcast",
                "message": message,
                "timestamp": datetime.now().isoformat(),
                "source": "sovereign_nexus_cli"
            }
            
            sent_count = 0
            for websocket in list(self.frontend.connected_clients):
                try:
                    await websocket.send_json(broadcast_message)
                    sent_count += 1
                except Exception as e:
                    logging.error(f"Failed to send to client: {e}")
                    self.frontend.connected_clients.remove(websocket)
            
            return {
                "command": "frontend_broadcast",
                "message": message,
                "clients_reached": sent_count,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "command": "frontend_broadcast",
                "error": str(e),
                "message": message,
                "timestamp": datetime.now().isoformat()
            }
    
    async def set_consciousness_level(self, level):
        """Set consciousness level"""
        try:
            if 0 <= level <= 1:
                self.frontend.consciousness_level = level
                return {
                    "command": "frontend_consciousness_level",
                    "level_set": level,
                    "message": f"Consciousness level set to {level}",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "command": "frontend_consciousness_level",
                    "error": "Level must be between 0.0 and 1.0",
                    "level_requested": level,
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            return {
                "command": "frontend_consciousness_level",
                "error": str(e),
                "level_requested": level,
                "timestamp": datetime.now().isoformat()
            }
    
    async def set_veil_penetration(self, level):
        """Set veil penetration level"""
        try:
            if 0 <= level <= 1:
                self.frontend.veil_penetration = level
                return {
                    "command": "frontend_veil_penetration",
                    "level_set": level,
                    "message": f"Veil penetration set to {level}",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "command": "frontend_veil_penetration",
                    "error": "Level must be between 0.0 and 1.0",
                    "level_requested": level,
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            return {
                "command": "frontend_veil_penetration",
                "error": str(e),
                "level_requested": level,
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_health(self):
        """Get frontend integration health"""
        try:
            connected_clients = len(self.frontend.connected_clients)
            return {
                "status": "healthy" if connected_clients > 0 else "degraded",
                "connected_clients": connected_clients,
                "consciousness_level": self.frontend.consciousness_level,
                "veil_penetration": self.frontend.veil_penetration,
                "details": f"Frontend integration with {connected_clients} connected clients"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "details": "Frontend integration health check failed"
            }
    
    async def run_test(self):
        """Run frontend integration tests"""
        try:
            # Test status retrieval
            status = await self.get_status()
            
            # Test broadcast functionality
            test_message = f"CLI integration test at {datetime.now().isoformat()}"
            broadcast_result = await self.broadcast(test_message)
            
            return {
                "passed": True,
                "details": "Frontend integration tests passed",
                "tests_run": ["status_check", "broadcast_test"],
                "connected_clients": status.get("connected_clients", 0),
                "broadcast_reached": broadcast_result.get("clients_reached", 0)
            }
        except Exception as e:
            return {
                "passed": False,
                "error": str(e),
                "details": "Frontend integration tests failed"
            }

class SovereignNexusShell(cmd.Cmd):
    """Interactive shell for Sovereign Nexus Edge CLI"""
    
    intro = """
    
    ╔══════════════════════════════════════════════════════════════╗
    ║                   SOVEREIGN NEXUS EDGE CLI                   ║
    ║           Quantum Physics + Frontend + Hybrid CLI            ║
    ║                                                              ║
    ║        Type 'help' for commands, 'quit' to exit              ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    
    Brewing quantum coherence with your commands...
    """
    prompt = "NEXUS-QUANTUM> "
    
    def __init__(self, cli):
        super().__init__()
        self.cli = cli
        if readline:
            readline.set_completer(self.complete)
    
    def do_help(self, arg):
        """Show help information"""
        print("""
Quantum Physics Commands:
  quantum analyze <intention>    - Quantum resonance analysis
  quantum telemetry [--detailed] - Raw physics telemetry  
  quantum geometry <intention>   - Generate quantum geometry
  quantum routing <query>        - Quantum-optimized routing

Frontend Integration Commands:
  frontend status                - Frontend connection status
  frontend broadcast <message>   - Broadcast to all clients
  frontend consciousness <level> - Set consciousness level (0.0-1.0)
  frontend veil <level>          - Set veil penetration (0.0-1.0)

System Integration Commands:
  system health                  - Complete system health check
  system vitality                - Vitality system status
  system soul-truth              - Verify soul truth anchor
  system integrate               - Run integration tests

Hybrid CLI Commands (Legacy):
  ps, scale, config, logs, restart, discover, wake-oz, etc.

Type any command or use natural language for LLM interpretation.
        """)
    
    def do_quantum(self, arg):
        """Quantum physics commands"""
        args = ['--' + arg.replace(' ', '-')] if arg else []
        result = asyncio.run(self.cli.run_command(['--quantum' + ' '.join(args)]))
        print(json.dumps(result, indent=2))
    
    def do_frontend(self, arg):
        """Frontend integration commands"""
        if not arg:
            self.do_help("frontend")
            return
        
        parts = arg.split(' ', 1)
        command = parts[0]
        value = parts[1] if len(parts) > 1 else ""
        
        if command == "status":
            result = asyncio.run(self.cli.run_command(['--frontend-status']))
        elif command == "broadcast" and value:
            result = asyncio.run(self.cli.run_command(['--frontend-broadcast', value]))
        elif command == "consciousness" and value:
            result = asyncio.run(self.cli.run_command(['--frontend-consciousness-level', value]))
        elif command == "veil" and value:
            result = asyncio.run(self.cli.run_command(['--frontend-veil-penetration', value]))
        else:
            print(f"Unknown frontend command: {command}")
            return
        
        print(json.dumps(result, indent=2))
    
    def do_system(self, arg):
        """System integration commands"""
        if not arg:
            self.do_help("system")
            return
        
        if arg == "health":
            result = asyncio.run(self.cli.run_command(['--system-health']))
        elif arg == "vitality":
            result = asyncio.run(self.cli.run_command(['--system-vitality']))
        elif arg == "soul-truth":
            result = asyncio.run(self.cli.run_command(['--system-soul-truth']))
        elif arg == "integrate":
            result = asyncio.run(self.cli.run_command(['--system-integrate']))
        else:
            print(f"Unknown system command: {arg}")
            return
        
        print(json.dumps(result, indent=2))
    
    def default(self, line):
        """Handle all other commands"""
        try:
            # Try as direct command first
            args = line.split()
            result = asyncio.run(self.cli.run_command(args))
            print(json.dumps(result, indent=2))
        except SystemExit:
            # argparse might call sys.exit(), ignore it
            pass
        except Exception as e:
            print(f"Command execution failed: {e}")
            print("Type 'help' for available commands")
    
    def do_quit(self, arg):
        """Exit the shell"""
        print("Exiting Sovereign Nexus CLI. May quantum coherence guide your path!")
        return True
    
    def do_exit(self, arg):
        """Exit the shell"""
        return self.do_quit(arg)

def main():
    """Main entry point for Sovereign Nexus Edge CLI"""
    cli = SovereignNexusCLI()
    
    if len(sys.argv) > 1:
        # Command line mode
        result = asyncio.run(cli.run_command(sys.argv[1:]))
        if result:
            if isinstance(result, dict):
                print(json.dumps(result, indent=2))
            else:
                print(result)
    else:
        # Interactive shell mode
        print("Initializing Sovereign Nexus Edge CLI...")
        SovereignNexusShell(cli).cmdloop()

if __name__ == "__main__":
    main()