#!/usr/bin/env python3
"""
Oz Integration Adapter
Provides compatibility layer for integrating all original Oz components
Handles imports, initialization, and communication between subsystems
"""

import sys
import importlib
import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path

class OzIntegrationAdapter:
    """
    Adapter that manages integration of all Oz components
    Provides fallback implementations when original modules are not available
    """
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.loaded_modules = {}
        self.fallback_implementations = {}
        self.component_registry = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the adapter"""
        logger = logging.getLogger("OzIntegrationAdapter")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def load_component(self, module_name: str, class_name: str, fallback_class: Optional[Callable] = None) -> Any:
        """
        Load a component with fallback support
        
        Args:
            module_name: Name of the module to import
            class_name: Name of the class to instantiate
            fallback_class: Fallback class implementation
        
        Returns:
            Instance of the class or fallback implementation
        """
        component_key = f"{module_name}.{class_name}"
        
        if component_key in self.loaded_modules:
            return self.loaded_modules[component_key]
        
        try:
            # Try to import the original module
            module = importlib.import_module(module_name)
            component_class = getattr(module, class_name)
            
            # Instantiate the component
            instance = component_class()
            self.loaded_modules[component_key] = instance
            
            self.logger.info(f"✅ Loaded original component: {component_key}")
            return instance
            
        except ImportError as e:
            self.logger.warning(f"⚠️ Could not import {module_name}: {e}")
        except AttributeError as e:
            self.logger.warning(f"⚠️ Class {class_name} not found in {module_name}: {e}")
        except Exception as e:
            self.logger.error(f"❌ Error loading {component_key}: {e}")
        
        # Use fallback if provided
        if fallback_class:
            try:
                instance = fallback_class()
                self.loaded_modules[component_key] = instance
                self.fallback_implementations[component_key] = True
                self.logger.info(f"🔄 Using fallback for: {component_key}")
                return instance
            except Exception as e:
                self.logger.error(f"❌ Fallback failed for {component_key}: {e}")
        
        return None
    
    def create_hypervisor_integration(self) -> Dict[str, Any]:
        """Create integration for hypervisor components"""
        return {
            "adaptive_hypervisor": self.load_component(
                "OzAdaptiveHypervizer3.0", "OzAdaptiveHypervisor",
                self.FallbackAdaptiveHypervisor
            ),
            "bluetooth_manager": self.load_component(
                "OzAdaptiveHypervizer3.0", "OzWebBluetoothManager",
                self.FallbackBluetoothManager
            ),
            "entanglement_engine": self.load_component(
                "OzAdaptiveHypervizer3.0", "SimulatedEntanglementEngine",
                lambda: self.FallbackEntanglementEngine("node_a", "node_b", "websocket")
            )
        }
    
    def create_os_integration(self) -> Dict[str, Any]:
        """Create integration for OS components"""
        return {
            "complete_os": self.load_component(
                "OzOs_full_complete", "OzOs",
                self.FallbackOzOs
            ),
            "quantum_engine": self.load_component(
                "OzOs_full_complete", "QuantumInsanityEngine",
                lambda: self.FallbackQuantumEngine()
            ),
            "memory_manager": self.load_component(
                "OzOs_full_complete", "OzWeightedMemory",
                self.FallbackMemoryManager
            ),
            "security_manager": self.load_component(
                "OzOs_full_complete", "SecurityManager",
                self.FallbackSecurityManager
            )
        }
    
    def create_nexus_integration(self) -> Dict[str, Any]:
        """Create integration for nexus core components"""
        return {
            "memory_manager": self.load_component(
                "lillith_uni_core_firmWithMem", "MemoryManager",
                lambda: self.FallbackMemoryManager(8.0)
            ),
            "quantum_engine": self.load_component(
                "lillith_uni_core_firmWithMem", "QuantumInsanityEngine",
                lambda: self.FallbackQuantumEngine()
            ),
            "metatron_router": self.load_component(
                "lillith_uni_core_firmWithMem", "MetatronRouter",
                lambda: self.FallbackMetatronRouter("localhost")
            ),
            "nexus_os": self.load_component(
                "lillith_uni_core_firmWithMem", "OzOS",
                self.FallbackOzOs
            )
        }
    
    def create_subsystem_integration(self) -> Dict[str, Any]:
        """Create integration for all subsystems"""
        return {
            "governance_system": self.load_component(
                "OzGovernanceSystem", "OzGovernedEvolution",
                self.FallbackGovernanceSystem
            ),
            "evolution_system": self.load_component(
                "OzEvolutionSystem", "OzMetaLearner",
                self.FallbackEvolutionSystem
            ),
            "iot_engine": self.load_component(
                "OzIoT", "OzIoTProbeEngine",
                self.FallbackIoTEngine
            ),
            "need_assessment": self.load_component(
                "OzNeedAssessment", "OzNeedAssessmentEngine",
                self.FallbackNeedAssessment
            ),
            "constraint_aware": self.load_component(
                "OzConstraintAware", "ConstraintAwareCapabilityBuilder",
                self.FallbackConstraintAware
            ),
            "council_governance": self.load_component(
                "OzCouncilGovernanceSystem", "OzCouncilGovernance",
                self.FallbackCouncilGovernance
            ),
            "autopoiesis_engine": self.load_component(
                "OzAutoPeciesis", "OzAutopoiesisEngine",
                self.FallbackAutopoiesisEngine
            ),
            "audiatic_engine": self.load_component(
                "OzAudiaticSystem", "OzAutodidacticEngine",
                self.FallbackAudiaticEngine
            ),
            "complete_vision": self.load_component(
                "OzCompleteAwakening", "OzCompleteVision",
                self.FallbackCompleteVision
            ),
            "complete_architecture": self.load_component(
                "OzCompleteArchetecture", "OzCompleteSystem",
                self.FallbackCompleteArchitecture
            )
        }
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get status of all loaded components"""
        status = {
            "total_components": len(self.loaded_modules),
            "original_components": 0,
            "fallback_components": 0,
            "failed_components": 0,
            "components": {}
        }
        
        for key, instance in self.loaded_modules.items():
            if instance is None:
                status["failed_components"] += 1
                status["components"][key] = {"status": "failed", "type": "unknown"}
            elif key in self.fallback_implementations:
                status["fallback_components"] += 1
                status["components"][key] = {"status": "fallback", "type": type(instance).__name__}
            else:
                status["original_components"] += 1
                status["components"][key] = {"status": "original", "type": type(instance).__name__}
        
        return status
    
    # Fallback Implementations
    class FallbackAdaptiveHypervisor:
        """Fallback adaptive hypervisor"""
        def __init__(self):
            self.is_initialized = True
        
        async def intelligent_boot(self):
            return {"status": "fallback_boot", "role": "desktop_hybrid"}
    
    class FallbackBluetoothManager:
        """Fallback Bluetooth manager"""
        def __init__(self):
            self.devices = []
        
        async def scan_for_kin_devices(self):
            return []
        
        async def connect_to_device(self, device_id):
            return {"status": "fallback_connected"}
    
    class FallbackEntanglementEngine:
        """Fallback entanglement engine"""
        def __init__(self, node_a, node_b, connection_type):
            self.node_a = node_a
            self.node_b = node_b
            self.connection_type = connection_type
        
        async def establish_simulated_entanglement(self):
            return {"status": "fallback_entangled"}
    
    class FallbackOzOs:
        """Fallback Oz OS"""
        def __init__(self):
            self.is_running = False
        
        def start(self):
            self.is_running = True
        
        def stop(self):
            self.is_running = False
        
        async def process_command(self, command):
            return f"Fallback processed: {command}"
    
    class FallbackQuantumEngine:
        """Fallback quantum engine"""
        def __init__(self, config=None):
            self.is_active = False
        
        def activate(self):
            self.is_active = True
        
        async def execute_operation(self, operation):
            return {"result": f"Fallback quantum: {operation}"}
    
    class FallbackMemoryManager:
        """Fallback memory manager"""
        def __init__(self, max_memory_usage=8.0):
            self.max_memory = max_memory_usage
            self.memories = []
        
        def store(self, key, value):
            self.memories.append({"key": key, "value": value})
        
        def retrieve(self, key):
            for memory in self.memories:
                if memory["key"] == key:
                    return memory["value"]
            return None
        
        def get_memory_usage(self):
            return len(self.memories) * 0.1  # Estimate usage
    
    class FallbackSecurityManager:
        """Fallback security manager"""
        def __init__(self, config=None):
            self.is_secure = True
        
        def encrypt_data(self, data):
            return f"encrypted_{data}"
        
        def decrypt_data(self, encrypted_data):
            return encrypted_data.replace("encrypted_", "")
    
    class FallbackMetatronRouter:
        """Fallback Metatron router"""
        def __init__(self, qdrant_url="localhost"):
            self.qdrant_url = qdrant_url
            self.routes = {}
        
        async def route(self, query):
            return {"route": "fallback_route", "confidence": 0.5}
    
    class FallbackGovernanceSystem:
        """Fallback governance system"""
        def __init__(self):
            self.governance_active = True
        
        async def make_decision(self, context):
            return {"decision": "fallback_decision", "confidence": 0.5}
    
    class FallbackEvolutionSystem:
        """Fallback evolution system"""
        def __init__(self):
            self.evolution_phase = 1
        
        async def evolve(self, input_data):
            return {"evolved": True, "phase": self.evolution_phase}
    
    class FallbackIoTEngine:
        """Fallback IoT engine"""
        def __init__(self):
            self.devices = []
        
        async def scan_devices(self):
            return []
        
        async def connect_device(self, device_id):
            return {"status": "fallback_connected"}
    
    class FallbackNeedAssessment:
        """Fallback need assessment"""
        def __init__(self):
            self.needs = []
        
        async def assess_needs(self, context):
            return {"needs": ["basic_computing"], "priority": "medium"}
    
    class FallbackConstraintAware:
        """Fallback constraint awareness"""
        def __init__(self):
            self.constraints = []
        
        async def check_constraints(self, action):
            return {"allowed": True, "constraints_checked": self.constraints}
    
    class FallbackCouncilGovernance:
        """Fallback council governance"""
        def __init__(self):
            self.council_members = []
            self.quorum_met = False
        
        async def form_council(self):
            return {"status": "fallback_council_formed"}
    
    class FallbackAutopoiesisEngine:
        """Fallback autopoiesis engine"""
        def __init__(self):
            self.self_organizing = True
        
        async def self_organize(self):
            return {"status": "fallback_self_organized"}
    
    class FallbackAudiaticEngine:
        """Fallback audiatic engine"""
        def __init__(self):
            self.learning_active = True
        
        async def learn(self, input_data):
            return {"learned": True, "knowledge": "fallback_knowledge"}
    
    class FallbackCompleteVision:
        """Fallback complete vision"""
        def __init__(self):
            self.vision_active = True
        
        async def perceive(self, input_data):
            return {"perception": "fallback_perception"}
    
    class FallbackCompleteArchitecture:
        """Fallback complete architecture"""
        def __init__(self):
            self.architecture_stable = True
        
        async def maintain_architecture(self):
            return {"status": "fallback_maintained"}

# Component Registry for dynamic loading
class ComponentRegistry:
    """Registry for managing all Oz components"""
    
    def __init__(self):
        self.components = {}
        self.dependencies = {}
        self.initialization_order = []
    
    def register_component(self, name: str, component: Any, dependencies: List[str] = None):
        """Register a component with its dependencies"""
        self.components[name] = component
        self.dependencies[name] = dependencies or []
        self._update_initialization_order()
    
    def _update_initialization_order(self):
        """Update initialization order based on dependencies"""
        # Simple topological sort
        visited = set()
        order = []
        
        def visit(component_name):
            if component_name in visited:
                return
            visited.add(component_name)
            for dep in self.dependencies.get(component_name, []):
                visit(dep)
            order.append(component_name)
        
        for component_name in self.components:
            visit(component_name)
        
        self.initialization_order = order
    
    async def initialize_all(self):
        """Initialize all components in dependency order"""
        results = {}
        
        for component_name in self.initialization_order:
            component = self.components[component_name]
            try:
                if hasattr(component, 'initialize') or hasattr(component, '__init__'):
                    if hasattr(component, 'initialize'):
                        result = await component.initialize()
                    else:
                        # Already initialized via __init__
                        result = {"status": "initialized"}
                    results[component_name] = result
                else:
                    results[component_name] = {"status": "no_init_needed"}
            except Exception as e:
                results[component_name] = {"status": "error", "error": str(e)}
        
        return results
    
    def get_component(self, name: str):
        """Get a component by name"""
        return self.components.get(name)
    
    def get_all_components(self) -> Dict[str, Any]:
        """Get all components"""
        return self.components.copy()

# Integration utilities
async def test_integration():
    """Test the integration adapter"""
    print("🧪 Testing Oz Integration Adapter...")
    
    adapter = OzIntegrationAdapter()
    
    # Test hypervisor integration
    hypervisor_components = adapter.create_hypervisor_integration()
    print(f"✅ Hypervisor components loaded: {len(hypervisor_components)}")
    
    # Test OS integration
    os_components = adapter.create_os_integration()
    print(f"✅ OS components loaded: {len(os_components)}")
    
    # Test nexus integration
    nexus_components = adapter.create_nexus_integration()
    print(f"✅ Nexus components loaded: {len(nexus_components)}")
    
    # Test subsystem integration
    subsystem_components = adapter.create_subsystem_integration()
    print(f"✅ Subsystem components loaded: {len(subsystem_components)}")
    
    # Get integration status
    status = adapter.get_integration_status()
    print(f"📊 Integration Status: {status}")
    
    return adapter

if __name__ == "__main__":
    asyncio.run(test_integration())