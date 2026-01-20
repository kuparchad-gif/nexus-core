import asyncio
import importlib
import logging
import os
import sys
import time
from typing import Dict, List, Any, Optional, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("ServiceManager")

class ServiceManager:
    """
    Manages the lifecycle of all system services
    """
    
    def __init__(self):
        self.services = {}
        self.service_status = {}
        self.dependencies = {}
        self.startup_order = []
        self.shutdown_order = []
        self.running = False
    
    def register_service(self, 
                        name: str, 
                        module_path: str, 
                        init_function: str = "initialize", 
                        dependencies: List[str] = None):
        """
        Register a service with the manager
        
        Args:
            name: Service name
            module_path: Import path to the service module
            init_function: Name of the initialization function
            dependencies: List of service names this service depends on
        """
        self.services[name] = {
            "name": name,
            "module_path": module_path,
            "init_function": init_function,
            "instance": None,
            "initialized": False
        }
        
        self.dependencies[name] = dependencies or []
        self.service_status[name] = "registered"
        
        logger.info(f"Registered service: {name} from {module_path}")
        
        # Recalculate startup and shutdown order
        self._calculate_startup_order()
    
    def _calculate_startup_order(self):
        """Calculate service startup and shutdown order based on dependencies"""
        # Simple topological sort
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(service_name):
            if service_name in temp_visited:
                raise ValueError(f"Circular dependency detected involving {service_name}")
            
            if service_name not in visited:
                temp_visited.add(service_name)
                
                for dep in self.dependencies.get(service_name, []):
                    if dep in self.services:
                        visit(dep)
                
                temp_visited.remove(service_name)
                visited.add(service_name)
                order.append(service_name)
        
        for service_name in self.services:
            if service_name not in visited:
                visit(service_name)
        
        self.startup_order = list(reversed(order))
        self.shutdown_order = order
        
        logger.info(f"Startup order: {self.startup_order}")
        logger.info(f"Shutdown order: {self.shutdown_order}")
    
    async def start_services(self):
        """Start all registered services in dependency order"""
        self.running = True
        
        logger.info("Starting services...")
        
        for service_name in self.startup_order:
            try:
                await self.start_service(service_name)
            except Exception as e:
                logger.error(f"Error starting service {service_name}: {e}")
                self.service_status[service_name] = f"error: {str(e)}"
        
        logger.info("All services started")
        return {"status": "started", "services": self.service_status}
    
    async def start_service(self, service_name: str):
        """Start a specific service"""
        if service_name not in self.services:
            raise ValueError(f"Service not registered: {service_name}")
        
        service = self.services[service_name]
        
        if service["initialized"]:
            logger.info(f"Service {service_name} already initialized")
            return
        
        # Check dependencies
        for dep_name in self.dependencies.get(service_name, []):
            if dep_name not in self.services:
                raise ValueError(f"Dependency {dep_name} not registered for {service_name}")
            
            dep_service = self.services[dep_name]
            if not dep_service["initialized"]:
                logger.info(f"Initializing dependency {dep_name} for {service_name}")
                await self.start_service(dep_name)
        
        # Import and initialize service
        logger.info(f"Starting service: {service_name}")
        self.service_status[service_name] = "starting"
        
        try:
            # Import module
            module = importlib.import_module(service["module_path"])
            
            # Get initialization function
            init_func = getattr(module, service["init_function"])
            
            # Initialize service
            result = init_func()
            
            # Handle async initialization
            if asyncio.iscoroutine(result):
                result = await result
            
            # Store result
            service["instance"] = result
            service["initialized"] = True
            self.service_status[service_name] = "running"
            
            logger.info(f"Service {service_name} started successfully")
            
        except Exception as e:
            logger.error(f"Error starting service {service_name}: {e}")
            self.service_status[service_name] = f"error: {str(e)}"
            raise
    
    async def stop_services(self):
        """Stop all services in reverse dependency order"""
        if not self.running:
            return {"status": "not_running"}
        
        logger.info("Stopping services...")
        
        for service_name in self.shutdown_order:
            try:
                await self.stop_service(service_name)
            except Exception as e:
                logger.error(f"Error stopping service {service_name}: {e}")
        
        self.running = False
        logger.info("All services stopped")
        
        return {"status": "stopped", "services": self.service_status}
    
    async def stop_service(self, service_name: str):
        """Stop a specific service"""
        if service_name not in self.services:
            raise ValueError(f"Service not registered: {service_name}")
        
        service = self.services[service_name]
        
        if not service["initialized"]:
            logger.info(f"Service {service_name} not running")
            return
        
        logger.info(f"Stopping service: {service_name}")
        self.service_status[service_name] = "stopping"
        
        try:
            # Get service instance
            instance = service["instance"]
            
            # Call stop method if available
            if hasattr(instance, "stop"):
                result = instance.stop()
                
                # Handle async stop
                if asyncio.iscoroutine(result):
                    await result
            
            service["initialized"] = False
            self.service_status[service_name] = "stopped"
            
            logger.info(f"Service {service_name} stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping service {service_name}: {e}")
            self.service_status[service_name] = f"error: {str(e)}"
            raise
    
    def get_service(self, service_name: str) -> Any:
        """Get a service instance by name"""
        if service_name not in self.services:
            raise ValueError(f"Service not registered: {service_name}")
        
        service = self.services[service_name]
        
        if not service["initialized"]:
            raise ValueError(f"Service {service_name} not initialized")
        
        return service["instance"]
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all services"""
        return {
            "running": self.running,
            "services": self.service_status,
            "startup_order": self.startup_order,
            "shutdown_order": self.shutdown_order
        }

# Singleton instance
_service_manager = None

def get_service_manager() -> ServiceManager:
    """Get or create the service manager singleton"""
    global _service_manager
    if _service_manager is None:
        _service_manager = ServiceManager()
    return _service_manager

def register_core_services():
    """Register all core system services"""
    manager = get_service_manager()
    
    # Register Heart service
    manager.register_service(
        name="heart",
        module_path="Services.Heart",
        init_function="initialize_heart_service",
        dependencies=[]
    )
    
    # Register Memory service
    manager.register_service(
        name="memory",
        module_path="Services.Memory",
        init_function="initialize_memory_service",
        dependencies=["heart"]
    )
    
    # Register other core services as needed
    # ...
    
    return manager

async def start_all_services():
    """Start all core services"""
    manager = register_core_services()
    return await manager.start_services()

async def stop_all_services():
    """Stop all services"""
    manager = get_service_manager()
    return await manager.stop_services()

if __name__ == "__main__":
    # When run directly, start all services
    asyncio.run(start_all_services())
