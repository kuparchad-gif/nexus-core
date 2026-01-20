# Services/system_status_checker.py

import os
import logging
import importlib
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Configure logging
logger = logging.getLogger("system_status_checker")
logger.setLevel(logging.INFO)
os.makedirs("logs", exist_ok=True)
handler = logging.FileHandler("logs/system_status.log")
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class SystemStatusChecker:
    """
    Checks the status of all critical Viren components.
    """
    
    def __init__(self):
        """Initialize the system status checker."""
        self.root_dir = Path(__file__).parent.parent
        self.critical_components = {
            "Processing Pipeline": ["Services/text_service.py", "Services/tone_service.py", "Systems/engine/core/signal/symbol_interpreter.py"],
            "Emotional Routing": ["Systems/planner/planner_service.py", "Services/runtime_loader.py"],
            "Pulse Timing": ["Systems/engine/pulse/pulse_core.py", "Systems/tone_mesh/modules/tone_sync_pulse.py"],
            "Security and Sync": ["Systems/security/trinity_gate/trinity_validator.py", "Systems/engine/guardian/guardian_core/guardian_service.py"],
            "Scalable Comms": ["Systems/engine/core/orchestrator.py", "bridge/mlx_bridge.py"],
            "Memory and Context": ["Systems/memory/memory_module.py", "Systems/memory/shard_manager.py", "Systems/memory/binary_emotion.py"],
            "Subconscious": ["Systems/Subconscious/modules/dream_stream.py", "Systems/Subconscious/modules/ego_stream.py", "Systems/mythrunner/mythrunner_manifest.yaml"],
            "Visual and Auditory": ["Services/vision_service.py", "Services/audio_service.py"],
            "Bias Detection": ["Systems/engine/core/signal/truth_patterning.py", "Systems/engine/core/signal/micro_pattern_profiling.py"],
            "Symbolic Reasoning": ["Systems/engine/core/signal/abstract_reasoning.py", "Systems/engine/core/signal/narrative_stack.py"]
        }
        
        # Add the new components
        self.new_components = {
            "Model Router": ["bridge/model_router.py"],
            "Self Management": ["Services/self_management.py", "Services/approval_system.py"],
            "Gradio MCP": ["Services/gradio_mcp.py"],
            "Technology Integrations": ["Services/technology_integrations.py"],
            "Viren Brain": ["Services/viren_brain.py"],
            "LoRA/QLoRA": ["models/lora"],
            "Transformers Agents": ["Services/advanced_integrations.py"],
            "LangChain/LiteChain": ["Services/advanced_integrations.py"],
            "DID (Decentralized ID)": ["Services/advanced_integrations.py"],
            "NeMo/DeepSpeed": ["Services/advanced_integrations.py"],
            "Diffusers + ControlNet": ["Services/advanced_integrations.py"],
            "Prometheus + OpenTelemetry": ["Services/advanced_integrations.py"],
            "LangGraph/Automata": ["Services/advanced_integrations.py"],
            "TorchMultimodal": ["Services/advanced_integrations.py"]
        }
        
        # Combine all components
        self.all_components = {**self.critical_components, **self.new_components}
    
    def check_component_status(self, component_name: str) -> Tuple[bool, List[str]]:
        """
        Check if a component is available and functioning.
        
        Args:
            component_name: Name of the component to check
            
        Returns:
            Tuple of (status, missing_files)
        """
        if component_name not in self.all_components:
            return False, [f"Unknown component: {component_name}"]
        
        files = self.all_components[component_name]
        missing_files = []
        
        for file_path in files:
            abs_path = os.path.join(self.root_dir, file_path)
            if not os.path.exists(abs_path):
                missing_files.append(file_path)
        
        return len(missing_files) == 0, missing_files
    
    def check_all_components(self) -> Dict[str, Dict[str, Any]]:
        """
        Check the status of all components.
        
        Returns:
            Dictionary with component status information
        """
        results = {}
        
        for component_name in self.all_components:
            status, missing_files = self.check_component_status(component_name)
            results[component_name] = {
                "status": "Active" if status else "Missing",
                "missing_files": missing_files,
                "critical": component_name in self.critical_components
            }
        
        return results
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get overall system health.
        
        Returns:
            Dictionary with system health information
        """
        component_status = self.check_all_components()
        
        # Count active and missing components
        total_components = len(component_status)
        active_components = sum(1 for comp in component_status.values() if comp["status"] == "Active")
        missing_components = total_components - active_components
        
        # Count critical components
        critical_components = sum(1 for comp in component_status.values() if comp["critical"])
        active_critical = sum(1 for comp in component_status.values() 
                             if comp["critical"] and comp["status"] == "Active")
        
        # Calculate health percentage
        health_percentage = (active_components / total_components) * 100
        critical_health_percentage = (active_critical / critical_components) * 100 if critical_components > 0 else 0
        
        # Determine overall status
        if critical_health_percentage == 100 and health_percentage >= 90:
            overall_status = "Excellent"
        elif critical_health_percentage >= 90 and health_percentage >= 75:
            overall_status = "Good"
        elif critical_health_percentage >= 75:
            overall_status = "Fair"
        else:
            overall_status = "Critical"
        
        return {
            "overall_status": overall_status,
            "health_percentage": health_percentage,
            "critical_health_percentage": critical_health_percentage,
            "active_components": active_components,
            "total_components": total_components,
            "active_critical": active_critical,
            "total_critical": critical_components,
            "component_status": component_status
        }
    
    def print_status_report(self):
        """Print a status report to the console."""
        health = self.get_system_health()
        
        print("\n===== LILLITH SYSTEM STATUS REPORT =====")
        print(f"Overall Status: {health['overall_status']}")
        print(f"System Health: {health['health_percentage']:.1f}%")
        print(f"Critical Systems: {health['critical_health_percentage']:.1f}%")
        print(f"Components: {health['active_components']}/{health['total_components']} active")
        print(f"Critical Components: {health['active_critical']}/{health['total_critical']} active")
        
        print("\n----- Component Status -----")
        for name, status in health['component_status'].items():
            status_str = status['status']
            critical_str = "[CRITICAL]" if status['critical'] else ""
            print(f"{name}: {status_str} {critical_str}")
            
            if status['status'] != "Active":
                print("  Missing files:")
                for file in status['missing_files']:
                    print(f"  - {file}")
        
        print("\n=====================================")
    
    def check_advanced_technologies(self):
        """Check the status of advanced technologies."""
        try:
            from Services.advanced_integrations import advanced_integrations
            
            print("\n===== ADVANCED TECHNOLOGIES STATUS =====")
            for tech, available in advanced_integrations.available_technologies.items():
                status = "Available" if available else "Not Available"
                print(f"{tech}: {status}")
            
            print("\n=====================================")
        except ImportError:
            print("Advanced integrations module not available")

# Create a singleton instance
system_checker = SystemStatusChecker()

# Example usage
if __name__ == "__main__":
    system_checker.print_status_report()
    system_checker.check_advanced_technologies()
