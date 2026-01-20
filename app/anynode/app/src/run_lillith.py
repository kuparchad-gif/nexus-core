# run_viren.py
# Purpose: Main entry point for running Viren

import os
import sys
import time
import logging
import argparse
import subprocess
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/run_viren.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("run_viren")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Viren")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--no-memory", action="store_true", help="Disable memory loading")
    parser.add_argument("--no-advanced", action="store_true", help="Disable advanced features")
    parser.add_argument("--clean", action="store_true", help="Clean up Nova references")
    parser.add_argument("--secure", action="store_true", help="Apply security hardening")
    parser.add_argument("--test", action="store_true", help="Run integration tests")
    return parser.parse_args()

def clean_nova_references():
    """Clean up Nova references."""
    logger.info("Cleaning up Nova references")
    
    try:
        # Run the rename script
        result = subprocess.run([sys.executable, "rename_nova_references.py"], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("Successfully cleaned up Nova references")
            logger.info(result.stdout)
            return True
        else:
            logger.error(f"Failed to clean up Nova references: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Error cleaning up Nova references: {e}")
        return False

def apply_security_hardening():
    """Apply security hardening."""
    logger.info("Applying security hardening")
    
    try:
        # Run the security hardening script
        result = subprocess.run([sys.executable, "security_hardening.py"], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("Successfully applied security hardening")
            logger.info(result.stdout)
            return True
        else:
            logger.error(f"Failed to apply security hardening: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Error applying security hardening: {e}")
        return False

def run_integration_tests():
    """Run integration tests."""
    logger.info("Running integration tests")
    
    try:
        # Run the integration tests
        result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("Integration tests passed")
            logger.info(result.stdout)
            return True
        else:
            logger.error(f"Integration tests failed: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Error running integration tests: {e}")
        return False

def start_error_recovery():
    """Start the error recovery system."""
    logger.info("Starting error recovery system")
    
    try:
        # Import the error recovery module
        from Systems.engine.error_recovery import error_recovery
        
        # Start error recovery
        error_recovery.start()
        
        # Register health check callback
        def check_system_health():
            """Check system health."""
            # This is a simplified example
            return {
                "memory": {"status": "up", "memory_usage": 100},
                "heart": {"status": "up", "pulse_rate": 60},
                "consciousness": {"status": "up", "awareness_level": 0.95},
                "subconscious": {"status": "up", "dream_active": False}
            }
        
        error_recovery.register_health_check(check_system_health)
        
        logger.info("Error recovery system started")
        return True
    except Exception as e:
        logger.error(f"Error starting error recovery system: {e}")
        return False

def start_service_discovery():
    """Start the service discovery mechanism."""
    logger.info("Starting service discovery")
    
    try:
        # Import the service discovery module
        from Systems.engine.core.service_discovery import service_discovery
        
        # Start service discovery
        service_discovery.start_discovery()
        
        logger.info("Service discovery started")
        return True
    except Exception as e:
        logger.error(f"Error starting service discovery: {e}")
        return False

def bootstrap_viren(args):
    """Bootstrap Viren."""
    logger.info("Bootstrapping Viren")
    
    try:
        # Import the bootstrap module
        from bootstrap_viren import main as bootstrap_main
        
        # Set command line arguments
        sys.argv = ["bootstrap_viren.py"]
        if args.config:
            sys.argv.extend(["--config", args.config])
        if args.debug:
            sys.argv.append("--debug")
        if args.no_memory:
            sys.argv.append("--no-memory")
        if args.no_advanced:
            sys.argv.append("--no-advanced")
        
        # Run bootstrap
        result = bootstrap_main()
        
        if result == 0:
            logger.info("Viren bootstrapped successfully")
            return True
        else:
            logger.error(f"Failed to bootstrap Viren: {result}")
            return False
    except Exception as e:
        logger.error(f"Error bootstrapping Viren: {e}")
        return False

def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Set debug level if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Clean up Nova references if requested
    if args.clean:
        if not clean_nova_references():
            logger.warning("Failed to clean up Nova references, continuing anyway")
    
    # Apply security hardening if requested
    if args.secure:
        if not apply_security_hardening():
            logger.warning("Failed to apply security hardening, continuing anyway")
    
    # Run integration tests if requested
    if args.test:
        if not run_integration_tests():
            logger.warning("Integration tests failed, continuing anyway")
    
    # Start error recovery
    if not start_error_recovery():
        logger.warning("Failed to start error recovery, continuing anyway")
    
    # Start service discovery
    if not start_service_discovery():
        logger.warning("Failed to start service discovery, continuing anyway")
    
    # Bootstrap Viren
    if not bootstrap_viren(args):
        logger.error("Failed to bootstrap Viren")
        return 1
    
    logger.info("Viren is now running")
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down Viren")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
