# firmware_orchestrator.py
import psutil
import asyncio
from datetime import datetime
from typing import Dict, List, Optional
import logging

class FirmwareOrchestrator:
    """Bridges CogniKube hardware with module transformation"""
    
    def __init__(self):
        self.hardware_metrics = {}
        self.module_registry = {}  # Tracks all transformable modules
        self.emergency_queue = []
        self.viren_deployment_queue = []
        self.transformation_history = []
        
        # Hardware thresholds (tuned for CogniKubes)
        self.thresholds = {
            "cpu_emergency": 80,    # Tag in modules immediately
            "cpu_warning": 70,      # Prepare for transformation
            "memory_emergency": 85, 
            "memory_warning": 75,
            "io_emergency": 90,
            "io_warning": 80
        }
        
        # Role transformation maps
        self.role_capabilities = {
            "language": ["memory", "reasoning", "light_compute"],
            "memory": ["language", "storage", "data_processing"], 
            "reasoning": ["language", "analysis", "light_compute"],
            "web": ["light_compute", "data_processing"],
            "vision": ["analysis", "data_processing"]
        }
        
        self.logger = logging.getLogger("FirmwareOrchestrator")
        
    async def start_hardware_monitoring(self):
        """Continuous hardware monitoring loop"""
        while True:
            await self._capture_hardware_metrics()
            await self._assess_system_health()
            await self._trigger_emergency_response()
            await asyncio.sleep(1)  # Real-time monitoring
    
    async def _capture_hardware_metrics(self):
        """Capture real CogniKube hardware metrics"""
        self.hardware_metrics = {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_io": self._get_disk_io_stats(),
            "network_io": self._get_network_io_stats(),
            "load_avg": psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0,
            "active_processes": len(psutil.pids())
        }
        
        # Log significant changes
        if self.hardware_metrics["cpu_percent"] > self.thresholds["cpu_warning"]:
            self.logger.warning(f"🚨 CPU at {self.hardware_metrics['cpu_percent']}%")
    
    async def _assess_system_health(self):
        """Assess if system needs module transformations"""
        health_status = "healthy"
        
        if self.hardware_metrics["cpu_percent"] > self.thresholds["cpu_emergency"]:
            health_status = "cpu_emergency"
            await self._queue_cpu_emergency()
            
        elif self.hardware_metrics["memory_percent"] > self.thresholds["memory_emergency"]:
            health_status = "memory_emergency" 
            await self._queue_memory_emergency()
            
        elif self.hardware_metrics["disk_io"]["busy_percent"] > self.thresholds["io_emergency"]:
            health_status = "io_emergency"
            await self._queue_io_emergency()
        
        return health_status
    
    async def _trigger_emergency_response(self):
        """Execute emergency transformations and deployments"""
        if not self.emergency_queue:
            return
            
        for emergency in self.emergency_queue[:3]:  # Process top 3 emergencies
            if await self._tag_in_existing_modules(emergency):
                self.logger.info(f"✅ Tagged in modules for {emergency['type']}")
            else:
                # No available modules - queue deployment via Viren
                await self._queue_viren_deployment(emergency)
                
        self.emergency_queue = []  # Clear processed emergencies
    
    async def _tag_in_existing_modules(self, emergency: Dict) -> bool:
        """Tag in existing modules for emergency support"""
        emergency_type = emergency["type"]
        required_role = self._map_emergency_to_role(emergency_type)
        
        # Find available modules that can transform to required role
        available_modules = []
        for module_id, module_data in self.module_registry.items():
            if (module_data["current_role"] != required_role and 
                required_role in self.role_capabilities.get(module_data["current_role"], []) and
                module_data["available_capacity"] > 0.3):  # At least 30% capacity
                
                available_modules.append((module_id, module_data))
        
        # Sort by capacity (highest first)
        available_modules.sort(key=lambda x: x[1]["available_capacity"], reverse=True)
        
        # Tag in modules (limit to 2 for stability)
        tagged_count = 0
        for module_id, module_data in available_modules[:2]:
            if await self._request_module_transformation(module_id, required_role, emergency):
                tagged_count += 1
                self.logger.info(f"🔄 Tagged in {module_id} as {required_role}")
        
        return tagged_count > 0
    
    async def _request_module_transformation(self, module_id: str, target_role: str, emergency: Dict) -> bool:
        """Request a module to transform to new role"""
        try:
            # This would call the module's transformation endpoint
            transformation_request = {
                "module_id": module_id,
                "target_role": target_role,
                "emergency_context": emergency,
                "priority": "firmware_emergency",
                "hardware_context": self.hardware_metrics
            }
            
            # Record transformation
            self.transformation_history.append({
                **transformation_request,
                "timestamp": datetime.now().isoformat(),
                "status": "requested"
            })
            
            # In real implementation, this would call module API
            # For now, simulate successful transformation
            self.module_registry[module_id]["current_role"] = target_role
            self.module_registry[module_id]["transformed_at"] = datetime.now().isoformat()
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Transformation failed for {module_id}: {e}")
            return False
    
    async def _queue_viren_deployment(self, emergency: Dict):
        """Queue new module deployment via Viren"""
        deployment_request = {
            "deployment_id": f"emergency_{datetime.now().strftime('%H%M%S')}",
            "emergency_type": emergency["type"],
            "required_role": self._map_emergency_to_role(emergency["type"]),
            "hardware_requirements": self._calculate_deployment_requirements(emergency),
            "priority": "firmware_emergency",
            "timestamp": datetime.now().isoformat()
        }
        
        self.viren_deployment_queue.append(deployment_request)
        self.logger.info(f"📋 Queued Viren deployment: {deployment_request['required_role']}")
    
    def _map_emergency_to_role(self, emergency_type: str) -> str:
        """Map hardware emergency to required module role"""
        emergency_role_map = {
            "cpu_emergency": "light_compute",
            "memory_emergency": "memory", 
            "io_emergency": "data_processing",
            "network_emergency": "light_compute"
        }
        return emergency_role_map.get(emergency_type, "light_compute")
    
    def _calculate_deployment_requirements(self, emergency: Dict) -> Dict:
        """Calculate hardware requirements for emergency deployment"""
        base_requirements = {
            "cpu_cores": 2,
            "memory_gb": 4,
            "storage_gb": 10,
            "network_priority": "high"
        }
        
        # Scale requirements based on emergency severity
        if emergency["type"] == "cpu_emergency":
            base_requirements["cpu_cores"] = 4
        elif emergency["type"] == "memory_emergency":
            base_requirements["memory_gb"] = 8
            
        return base_requirements
    
    def _get_disk_io_stats(self) -> Dict:
        """Get disk I/O statistics"""
        try:
            io_counters = psutil.disk_io_counters()
            return {
                "read_mb": io_counters.read_bytes / 1024 / 1024 if io_counters else 0,
                "write_mb": io_counters.write_bytes / 1024 / 1024 if io_counters else 0,
                "busy_percent": io_counters.busy_time / 1000 if io_counters else 0  # Simplified
            }
        except:
            return {"read_mb": 0, "write_mb": 0, "busy_percent": 0}
    
    def _get_network_io_stats(self) -> Dict:
        """Get network I/O statistics"""
        try:
            net_io = psutil.net_io_counters()
            return {
                "bytes_sent_mb": net_io.bytes_sent / 1024 / 1024,
                "bytes_recv_mb": net_io.bytes_recv / 1024 / 1024
            }
        except:
            return {"bytes_sent_mb": 0, "bytes_recv_mb": 0}
    
    # Module Registration
    def register_module(self, module_id: str, capabilities: List[str], current_role: str):
        """Register a transformable module with firmware"""
        self.module_registry[module_id] = {
            "current_role": current_role,
            "capabilities": capabilities,
            "available_capacity": 1.0,  # 0.0 to 1.0 scale
            "registered_at": datetime.now().isoformat(),
            "last_heartbeat": datetime.now().isoformat()
        }
        self.logger.info(f"📝 Registered module {module_id} as {current_role}")
    
    def update_module_capacity(self, module_id: str, capacity: float):
        """Update module's available capacity"""
        if module_id in self.module_registry:
            self.module_registry[module_id]["available_capacity"] = capacity
            self.module_registry[module_id]["last_heartbeat"] = datetime.now().isoformat()
    
    # Emergency Queue Management
    async def _queue_cpu_emergency(self):
        """Queue CPU emergency response"""
        self.emergency_queue.append({
            "type": "cpu_emergency",
            "severity": self.hardware_metrics["cpu_percent"] / 100,
            "timestamp": datetime.now().isoformat()
        })
    
    async def _queue_memory_emergency(self):
        """Queue memory emergency response"""
        self.emergency_queue.append({
            "type": "memory_emergency", 
            "severity": self.hardware_metrics["memory_percent"] / 100,
            "timestamp": datetime.now().isoformat()
        })
    
    async def _queue_io_emergency(self):
        """Queue I/O emergency response"""
        self.emergency_queue.append({
            "type": "io_emergency",
            "severity": self.hardware_metrics["disk_io"]["busy_percent"] / 100,
            "timestamp": datetime.now().isoformat()
        })

# Viren Deployment Processor
class VirenDeploymentProcessor:
    """Processes firmware deployment queues"""
    
    def __init__(self, firmware_orchestrator):
        self.firmware = firmware_orchestrator
        self.logger = logging.getLogger("VirenDeployment")
    
    async def process_deployment_queue(self):
        """Continuously process deployment queue"""
        while True:
            if self.firmware.viren_deployment_queue:
                deployment = self.firmware.viren_deployment_queue.pop(0)
                await self._deploy_emergency_module(deployment)
            
            await asyncio.sleep(5)  # Check queue every 5 seconds
    
    async def _deploy_emergency_module(self, deployment: Dict):
        """Deploy emergency module via Viren"""
        self.logger.info(f"🩺 Viren deploying: {deployment['required_role']}")
        
        # In real implementation, this would call Viren's deployment system
        # For now, simulate deployment
        new_module_id = f"emergency_{deployment['required_role']}_{datetime.now().strftime('%H%M%S')}"
        
        # Register new module with firmware
        self.firmware.register_module(
            module_id=new_module_id,
            capabilities=[deployment["required_role"]],
            current_role=deployment["required_role"]
        )
        
        self.logger.info(f"✅ Viren deployed {new_module_id} for {deployment['emergency_type']}")

# Integration with Existing System
async def main():
    """Integrate firmware orchestration with existing system"""
    firmware = FirmwareOrchestrator()
    viren_processor = VirenDeploymentProcessor(firmware)
    
    # Start monitoring and processing
    asyncio.create_task(firmware.start_hardware_monitoring())
    asyncio.create_task(viren_processor.process_deployment_queue())
    
    # Register existing modules (example)
    firmware.register_module("language_module_1", ["language", "memory", "reasoning"], "language")
    firmware.register_module("memory_module_1", ["memory", "storage", "data_processing"], "memory")
    firmware.register_module("web_module_1", ["web", "light_compute"], "web")
    
    print("🚀 Firmware Orchestration Active - Monitoring CogniKubes & Managing Transformations")
    
    # Keep running
    while True:
        await asyncio.sleep(60)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())