#!/usr/bin/env python3
"""
Boot Sequence for Viren
Main entry point for booting the Viren system
"""

import os
import sys
import time
import logging
import argparse
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('C:/Viren/logs', 'boot.log'))
    ]
)
logger = logging.getLogger("BootSequence")

def boot_viren(system_type: str = "desktop") -> bool:
    """Boot the Viren system"""
    logger.info(f"Starting Viren boot sequence for {system_type} system")
    
    try:
        # Import boot manager
        from core.boot_manager import BootManager
        
        # Create boot manager
        boot_manager = BootManager()
        
        # Start boot sequence
        success = boot_manager.boot()
        
        if success:
            logger.info(f"Viren {system_type} system booted successfully")
            
            # Get boot status
            all_required_running, status = boot_manager.verify_boot()
            
            if all_required_running:
                logger.info("All required components are running")
            else:
                logger.warning("Some required components failed to start")
                for component, component_status in status.items():
                    logger.info(f"Component {component}: {component_status}")
            
            return all_required_running
        else:
            logger.error(f"Failed to boot Viren {system_type} system")
            return False
    
    except Exception as e:
        logger.error(f"Error during boot sequence: {e}")
        return False

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Viren Boot Sequence")
    parser.add_argument("--system-type", type=str, default="desktop", choices=["desktop", "cloud", "portable"],
                        help="Type of Viren system to boot")
    
    args = parser.parse_args()
    
    # Set environment variable for system type
    os.environ["VIREN_SYSTEM_TYPE"] = args.system_type
    
    # Boot Viren
    success = boot_viren(args.system_type)
    
    if success:
        # Import system integrator
        try:
            from core.system_integrator import SystemIntegrator
            from core.database_registry import DatabaseRegistry
            
            # Initialize database registry
            database_registry = DatabaseRegistry()
            
            # Initialize system integrator
            system_integrator = SystemIntegrator(database_registry=database_registry)
            system_integrator.initialize()
            
            # Sync with remote systems
            logger.info("Syncing with remote systems...")
            sync_result = system_integrator.sync_with_remote_systems()
            
            if sync_result.get("success"):
                logger.info(f"Synced with {len(sync_result.get('synced_with', []))} remote systems")
            else:
                logger.warning(f"Sync with remote systems failed: {sync_result.get('error')}")
            
            # Sync Soulseed if available
            logger.info("Syncing Soulseed...")
            soulseed_result = system_integrator.sync_soulseed()
            
            if soulseed_result.get("success"):
                logger.info("Soulseed synced successfully")
            else:
                logger.warning(f"Soulseed sync failed: {soulseed_result.get('error')}")
        
        except Exception as e:
            logger.error(f"Error initializing system integrator: {e}")
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()