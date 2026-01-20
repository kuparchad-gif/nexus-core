#!/usr/bin/env python3
"""
Database Integration for Cloud Viren
Integrates database access into all Viren components
"""

import os
import sys
import importlib
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DatabaseIntegration")

# Import database manager
try:
    from database_manager import DatabaseManager
except ImportError:
    logger.error("Database manager not found. Make sure database_manager.py is in the correct location.")
    sys.exit(1)

class DatabaseIntegration:
    """
    Database Integration for Cloud Viren
    Integrates database access into all Viren components
    """
    
    def __init__(self):
        """Initialize the database integration"""
        self.db = DatabaseManager()
        self.integrated_components = {}
        
        logger.info("Database integration initialized")
    
    def integrate_component(self, component_name: str, component_module: str) -> bool:
        """
        Integrate database access into a component
        
        Args:
            component_name: Name of the component
            component_module: Module name of the component
            
        Returns:
            True if integration was successful, False otherwise
        """
        logger.info(f"Integrating database access into {component_name}")
        
        try:
            # Import the component module
            module = importlib.import_module(component_module)
            
            # Check if the component has a main class
            main_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and attr.__module__ == component_module:
                    # Found a class defined in this module
                    main_class = attr
                    break
            
            if not main_class:
                logger.error(f"No main class found in {component_module}")
                return False
            
            # Check if the component already has database integration
            if hasattr(main_class, 'db'):
                logger.warning(f"Component {component_name} already has database integration")
                return True
            
            # Add database access to the component
            setattr(main_class, 'db', self.db)
            
            # Add database methods to the component
            self._add_database_methods(main_class)
            
            # Record the integration
            self.integrated_components[component_name] = {
                "module": component_module,
                "class": main_class.__name__,
                "timestamp": import_time.time()
            }
            
            logger.info(f"Successfully integrated database access into {component_name}")
            return True
        
        except Exception as e:
            logger.error(f"Error integrating database access into {component_name}: {e}")
            return False
    
    def _add_database_methods(self, component_class) -> None:
        """Add database methods to a component class"""
        # Add save_to_db method
        def save_to_db(self, table: str, data: Dict[str, Any]) -> int:
            """Save data to the database"""
            if not hasattr(self, 'db'):
                raise AttributeError("Component does not have database access")
            
            return self.db.insert(table, data)
        
        setattr(component_class, 'save_to_db', save_to_db)
        
        # Add update_in_db method
        def update_in_db(self, table: str, data: Dict[str, Any], where: str, where_params: tuple) -> int:
            """Update data in the database"""
            if not hasattr(self, 'db'):
                raise AttributeError("Component does not have database access")
            
            return self.db.update(table, data, where, where_params)
        
        setattr(component_class, 'update_in_db', update_in_db)
        
        # Add get_from_db method
        def get_from_db(self, table: str, columns: str = "*", where: str = None, where_params: tuple = None,
                      order_by: str = None, limit: int = None, offset: int = None) -> list:
            """Get data from the database"""
            if not hasattr(self, 'db'):
                raise AttributeError("Component does not have database access")
            
            return self.db.select(table, columns, where, where_params, order_by, limit, offset)
        
        setattr(component_class, 'get_from_db', get_from_db)
    
    def integrate_all_components(self) -> Dict[str, bool]:
        """Integrate database access into all Viren components"""
        logger.info("Integrating database access into all components")
        
        # Define components to integrate
        components = {
            "diagnostic_core": "diagnostic_core",
            "research_tentacles": "research_tentacles",
            "blockchain_relay": "blockchain_relay",
            "llm_client": "llm_client",
            "cloud_connection": "cloud_connection",
            "lillith_heart_monitor": "lillith_heart_monitor",
            "model_cascade_manager": "model_cascade_manager",
            "modal_deployment": "modal_deployment"
        }
        
        # Integrate each component
        results = {}
        for component_name, component_module in components.items():
            results[component_name] = self.integrate_component(component_name, component_module)
        
        return results
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get integration status"""
        return {
            "integrated_components": self.integrated_components,
            "database_status": self.db.get_status()
        }

# Example usage
if __name__ == "__main__":
    import time as import_time
    
    # Create database integration
    integration = DatabaseIntegration()
    
    # Integrate all components
    results = integration.integrate_all_components()
    
    # Print results
    for component_name, success in results.items():
        print(f"Integration of {component_name}: {'Success' if success else 'Failed'}")
    
    # Print integration status
    status = integration.get_integration_status()
    print(f"Integrated components: {len(status['integrated_components'])}")
    print(f"Database tables: {status['database_status']['tables']}")
    print(f"Database size: {status['database_status']['size']} bytes")