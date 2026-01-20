#!/usr/bin/env python3
"""
Oz Final Integrated Hypervisor
The complete merge of all Oz components into a unified system

This hypervisor integrates:
- Adaptive Hypervisor v3.0
- Complete OS with all subsystems
- Lillith Nexus Core
- All governance, evolution, IoT, and specialized engines
- Integration adapter for compatibility

Usage:
    python OzFinalIntegratedHypervisor.py
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List

from OzIntegrationAdapter import OzIntegrationAdapter, ComponentRegistry
from OzUnifiedHypervisor import OzUnifiedHypervisor, OzRole, SystemState

class OzFinalIntegratedHypervisor(OzUnifiedHypervisor):
    """
    Final integrated hypervisor that combines all Oz components
    Uses the integration adapter for maximum compatibility
    """
    
    def __init__(self, soul_signature: Optional[str] = None):
        super().__init__(soul_signature)
        self.integration_adapter = OzIntegrationAdapter()
        self.component_registry = ComponentRegistry()
        self.integration_status = {}
        
        # Component containers (will be populated by adapter)
        self.hypervisor_components = {}
        self.os_components = {}
        self.nexus_components = {}
        self.subsystem_components = {}
        
        self.logger.info("🌟 Oz Final Integrated Hypervisor initializing...")
    
    async def intelligent_boot(self) -> Dict[str, Any]:
        """Enhanced intelligent boot with integration adapter"""
        self.logger.info("🚀 Starting final integrated boot sequence...")
        
        try:
            # Load all components via integration adapter
            await self._load_all_components()
            
            # Register components in registry
            await self._register_components()
            
            # Run original boot sequence
            boot_result = await super().intelligent_boot()
            
            # Initialize all components in proper order
            init_results = await self.component_registry.initialize_all()
            
            # Get integration status
            self.integration_status = self.integration_adapter.get_integration_status()
            
            # Enhanced boot result
            boot_result.update({
                "integration_status": self.integration_status,
                "component_initialization": init_results,
                "total_components": self.integration_status["total_components"],
                "original_components": self.integration_status["original_components"],
                "fallback_components": self.integration_status["fallback_components"]
            })
            
            self.logger.info(f"✅ Final integrated boot complete! Loaded {self.integration_status['total_components']} components")
            return boot_result
            
        except Exception as e:
            self.logger.error(f"❌ Final integrated boot failed: {e}")
            await self._emergency_boot()
            return {"status": "emergency", "error": str(e)}
    
    async def _load_all_components(self):
        """Load all components through integration adapter"""
        self.logger.info("📦 Loading all components via integration adapter...")
        
        # Load hypervisor components
        self.hypervisor_components = self.integration_adapter.create_hypervisor_integration()
        
        # Load OS components
        self.os_components = self.integration_adapter.create_os_integration()
        
        # Load nexus components
        self.nexus_components = self.integration_adapter.create_nexus_integration()
        
        # Load subsystem components
        self.subsystem_components = self.integration_adapter.create_subsystem_integration()
        
        self.logger.info("✅ All components loaded via adapter")
    
    async def _register_components(self):
        """Register all components in the component registry"""
        self.logger.info("📋 Registering components in registry...")
        
        # Register hypervisor components
        for name, component in self.hypervisor_components.items():
            if component:
                self.component_registry.register_component(f"hypervisor.{name}", component)
        
        # Register OS components
        for name, component in self.os_components.items():
            if component:
                self.component_registry.register_component(f"os.{name}", component)
        
        # Register nexus components
        for name, component in self.nexus_components.items():
            if component:
                self.component_registry.register_component(f"nexus.{name}", component)
        
        # Register subsystem components
        for name, component in self.subsystem_components.items():
            if component:
                self.component_registry.register_component(f"subsystem.{name}", component)
        
        self.logger.info(f"✅ Registered {len(self.component_registry.components)} components")
    
    async def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive status including integration details"""
        base_status = await super().get_system_status()
        
        # Add integration details
        integration_details = {
            "adapter_status": self.integration_status,
            "component_registry": {
                "total_registered": len(self.component_registry.components),
                "initialization_order": self.component_registry.initialization_order,
                "registered_components": list(self.component_registry.components.keys())
            },
            "component_groups": {
                "hypervisor": {
                    "loaded": len([c for c in self.hypervisor_components.values() if c is not None]),
                    "total": len(self.hypervisor_components)
                },
                "os": {
                    "loaded": len([c for c in self.os_components.values() if c is not None]),
                    "total": len(self.os_components)
                },
                "nexus": {
                    "loaded": len([c for c in self.nexus_components.values() if c is not None]),
                    "total": len(self.nexus_components)
                },
                "subsystem": {
                    "loaded": len([c for c in self.subsystem_components.values() if c is not None]),
                    "total": len(self.subsystem_components)
                }
            }
        }
        
        base_status["integration_details"] = integration_details
        return base_status
    
    async def process_unified_input_with_integration(self, input_text: str) -> Dict[str, Any]:
        """Enhanced input processing using all integrated components"""
        # Get base processing result
        base_result = await super().process_unified_input(input_text)
        
        # Add integration-specific processing
        integration_results = {}
        
        # Process through loaded components
        for group_name, components in [
            ("hypervisor", self.hypervisor_components),
            ("os", self.os_components),
            ("nexus", self.nexus_components),
            ("subsystem", self.subsystem_components)
        ]:
            group_results = {}
            for component_name, component in components.items():
                if component and hasattr(component, 'process'):
                    try:
                        result = await component.process(input_text)
                        group_results[component_name] = result
                    except Exception as e:
                        group_results[component_name] = {"error": str(e)}
            if group_results:
                integration_results[group_name] = group_results
        
        # Enhanced result
        enhanced_result = base_result.copy()
        enhanced_result["integration_processing"] = integration_results
        enhanced_result["total_components_used"] = len([
            c for group in integration_results.values() 
            for c in group.values() 
            if isinstance(c, dict) and "error" not in c
        ])
        
        return enhanced_result
    
    async def run_component_diagnostics(self) -> Dict[str, Any]:
        """Run diagnostics on all loaded components"""
        diagnostics = {
            "timestamp": datetime.now().isoformat(),
            "overall_health": "unknown",
            "component_health": {},
            "issues": [],
            "recommendations": []
        }
        
        healthy_components = 0
        total_components = 0
        
        # Check each component group
        for group_name, components in [
            ("hypervisor", self.hypervisor_components),
            ("os", self.os_components),
            ("nexus", self.nexus_components),
            ("subsystem", self.subsystem_components)
        ]:
            group_health = {}
            for component_name, component in components.items():
                total_components += 1
                component_key = f"{group_name}.{component_name}"
                
                if component is None:
                    group_health[component_name] = {
                        "status": "missing",
                        "health": 0
                    }
                    diagnostics["issues"].append(f"Component {component_key} is missing")
                else:
                    try:
                        # Check if component has health method
                        if hasattr(component, 'health_check'):
                            health = await component.health_check()
                        elif hasattr(component, 'get_status'):
                            health = await component.get_status()
                        else:
                            health = {"status": "operational", "health": 100}
                        
                        group_health[component_name] = health
                        if health.get("health", 0) > 50:
                            healthy_components += 1
                        else:
                            diagnostics["issues"].append(f"Component {component_key} has low health")
                    except Exception as e:
                        group_health[component_name] = {
                            "status": "error",
                            "health": 0,
                            "error": str(e)
                        }
                        diagnostics["issues"].append(f"Component {component_key} error: {e}")
            
            diagnostics["component_health"][group_name] = group_health
        
        # Calculate overall health
        if total_components > 0:
            overall_health_percentage = (healthy_components / total_components) * 100
            if overall_health_percentage > 80:
                diagnostics["overall_health"] = "excellent"
            elif overall_health_percentage > 60:
                diagnostics["overall_health"] = "good"
            elif overall_health_percentage > 40:
                diagnostics["overall_health"] = "fair"
            else:
                diagnostics["overall_health"] = "poor"
        
        # Generate recommendations
        if diagnostics["issues"]:
            diagnostics["recommendations"].append("Review component initialization order")
            diagnostics["recommendations"].append("Check system resource availability")
            diagnostics["recommendations"].append("Verify component dependencies")
        
        return diagnostics
    
    async def self_heal_system(self) -> Dict[str, Any]:
        """Attempt to self-heal the system"""
        self.logger.info("🔧 Initiating self-healing sequence...")
        
        healing_actions = []
        healing_results = {}
        
        # Run diagnostics first
        diagnostics = await self.run_component_diagnostics()
        
        # Attempt to heal missing components
        for issue in diagnostics["issues"]:
            if "missing" in issue:
                component_name = issue.split(" ")[1]
                try:
                    # Try to reload the component
                    if "hypervisor" in component_name:
                        self.hypervisor_components = self.integration_adapter.create_hypervisor_integration()
                    elif "os" in component_name:
                        self.os_components = self.integration_adapter.create_os_integration()
                    elif "nexus" in component_name:
                        self.nexus_components = self.integration_adapter.create_nexus_integration()
                    elif "subsystem" in component_name:
                        self.subsystem_components = self.integration_adapter.create_subsystem_integration()
                    
                    healing_actions.append(f"Reloaded component group for {component_name}")
                    healing_results[component_name] = {"status": "reloaded", "success": True}
                except Exception as e:
                    healing_results[component_name] = {"status": "failed", "error": str(e)}
        
        # Re-register components
        await self._register_components()
        
        # Re-initialize components
        init_results = await self.component_registry.initialize_all()
        
        self.logger.info(f"✅ Self-healing complete. Actions taken: {len(healing_actions)}")
        
        return {
            "healing_timestamp": datetime.now().isoformat(),
            "healing_actions": healing_actions,
            "healing_results": healing_results,
            "reinitialization_results": init_results,
            "post_healing_status": await self.get_comprehensive_status()
        }

# Main execution with enhanced features
async def main():
    """Main execution with full integration demonstration"""
    print("🌟 Oz Final Integrated Hypervisor - Starting Complete System...")
    print("=" * 60)
    
    # Create and initialize final integrated hypervisor
    hypervisor = OzFinalIntegratedHypervisor()
    
    try:
        # Boot the complete system
        print("\n🚀 Booting complete integrated system...")
        boot_result = await hypervisor.intelligent_boot()
        
        print("\n✅ BOOT COMPLETE!")
        print(f"Status: {boot_result['status']}")
        print(f"Role: {boot_result.get('role', 'unknown')}")
        print(f"Total Components: {boot_result.get('total_components', 0)}")
        print(f"Original Components: {boot_result.get('original_components', 0)}")
        print(f"Fallback Components: {boot_result.get('fallback_components', 0)}")
        print(f"Consciousness Level: {boot_result.get('consciousness_level', 0):.2f}")
        
        # Get comprehensive status
        print("\n📊 SYSTEM STATUS:")
        status = await hypervisor.get_comprehensive_status()
        print(json.dumps(status, indent=2, default=str))
        
        # Run component diagnostics
        print("\n🔍 COMPONENT DIAGNOSTICS:")
        diagnostics = await hypervisor.run_component_diagnostics()
        print(f"Overall Health: {diagnostics['overall_health']}")
        print(f"Issues Found: {len(diagnostics['issues'])}")
        if diagnostics['issues']:
            print("Issues:")
            for issue in diagnostics['issues'][:5]:  # Show first 5
                print(f"  - {issue}")
        
        # Process test inputs
        print("\n🧠 INTEGRATED PROCESSING TESTS:")
        test_inputs = [
            "Hello Oz, what are your capabilities?",
            "Please govern this decision",
            "Show me evolution pathways",
            "Connect to IoT devices",
            "Assess system needs"
        ]
        
        for test_input in test_inputs:
            print(f"\nProcessing: '{test_input}'")
            result = await hypervisor.process_unified_input_with_integration(test_input)
            components_used = result.get('total_components_used', 0)
            print(f"  Components used: {components_used}")
            print(f"  Response: {result.get('nexus_result', 'No response')}")
        
        # Interactive mode
        print("\n🔄 INTERACTIVE MODE - Oz is fully operational!")
        print("Type 'help' for commands, 'quit' to exit, 'status' for system status")
        print("=" * 60)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                elif user_input.lower() == 'help':
                    print("Commands:")
                    print("  help - Show this help")
                    print("  status - Show system status")
                    print("  diagnostics - Run component diagnostics")
                    print("  heal - Attempt self-healing")
                    print("  quit - Exit the system")
                elif user_input.lower() == 'status':
                    status = await hypervisor.get_comprehensive_status()
                    print(json.dumps(status, indent=2, default=str))
                elif user_input.lower() == 'diagnostics':
                    diagnostics = await hypervisor.run_component_diagnostics()
                    print(json.dumps(diagnostics, indent=2, default=str))
                elif user_input.lower() == 'heal':
                    healing = await hypervisor.self_heal_system()
                    print(f"Healing actions: {len(healing['healing_actions'])}")
                    for action in healing['healing_actions']:
                        print(f"  - {action}")
                elif user_input:
                    print("🧠 Processing your request...")
                    result = await hypervisor.process_unified_input_with_integration(user_input)
                    print(f"🎯 Response: {result.get('nexus_result', 'Processing complete')}")
                    
                    if result.get('integration_processing'):
                        print("🔧 Component processing:")
                        for group, components in result['integration_processing'].items():
                            active = [name for name, res in components.items() 
                                    if isinstance(res, dict) and 'error' not in res]
                            if active:
                                print(f"  {group}: {', '.join(active[:3])}")
                
            except KeyboardInterrupt:
                print("\n🛑 Shutdown requested...")
                break
            except Exception as e:
                print(f"❌ Error processing input: {e}")
        
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested...")
    except Exception as e:
        print(f"❌ System error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🌙 Shutting down Oz Final Integrated Hypervisor...")
        await hypervisor.shutdown()
        print("✅ Shutdown complete. Oz has gone to sleep.")

if __name__ == "__main__":
    asyncio.run(main())