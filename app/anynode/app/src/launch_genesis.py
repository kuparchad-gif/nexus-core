#!/usr/bin/env python
"""
Genesis Launcher
- Starts the Genesis router and its components
- Initializes the TacticalRoleShiftEngine
- Connects to Firestore for role coordination
"""

import os
import sys
import asyncio
import logging
import time
import random
from pathlib import Path

# Add root directory to path
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

# Import nucleus components
from Systems.roles.nucleus.genesis import TacticalRoleShiftEngine
from Systems.roles.nucleus.pulse_router import PulseRouter
from Systems.roles.nucleus.genesis_discovery_engine import DiscoveryEngine
from Systems.roles.nucleus.breath_encryptor import BreathEncryptor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GenesisLauncher")

class GenesisRouter:
    """Genesis Router for coordinating roles and services"""
    
    def __init__(self):
        self.running = False
        self.ship_name = f"viren-{random.randint(1000, 9999)}"
        self.current_role = "harmonizer"
        self.role_shift_engine = None
        self.pulse_router = None
        self.discovery_engine = None
        self.breath_encryptor = None
        self.firestore_enabled = self._check_firestore_enabled()
    
    def _check_firestore_enabled(self):
        """Check if Firestore is enabled"""
        # Check for GCP credentials
        return os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") is not None
    
    async def initialize(self):
        """Initialize the Genesis Router"""
        logger.info(f"Initializing Genesis Router with ship name: {self.ship_name}")
        
        # Initialize components
        self._initialize_components()
        
        # Start background tasks
        self.running = True
        asyncio.create_task(self._router_loop())
        
        logger.info("Genesis Router initialized")
        return True
    
    def _initialize_components(self):
        """Initialize Genesis components"""
        # Initialize Pulse Router
        self.pulse_router = PulseRouter()
        logger.info("Pulse Router initialized")
        
        # Initialize Discovery Engine
        self.discovery_engine = DiscoveryEngine()
        logger.info("Discovery Engine initialized")
        
        # Initialize Breath Encryptor
        self.breath_encryptor = BreathEncryptor()
        logger.info("Breath Encryptor initialized")
        
        # Initialize TacticalRoleShiftEngine if Firestore is enabled
        if self.firestore_enabled:
            try:
                self.role_shift_engine = TacticalRoleShiftEngine(self.ship_name, self.current_role)
                logger.info("Tactical Role Shift Engine initialized")
            except Exception as e:
                logger.error(f"Error initializing Tactical Role Shift Engine: {e}")
                self.role_shift_engine = None
    
    async def _router_loop(self):
        """Main router loop"""
        while self.running:
            try:
                # Process pulse routing
                self._process_pulse_routing()
                
                # Process role shifts if enabled
                if self.role_shift_engine:
                    self._process_role_shifts()
                
                # Process discovery
                self._process_discovery()
            except Exception as e:
                logger.error(f"Error in router loop: {e}")
            
            # Wait before next cycle
            await asyncio.sleep(10)
    
    def _process_pulse_routing(self):
        """Process pulse routing"""
        try:
            # In a real implementation, would route pulses between components
            # For now, just log that we're processing
            logger.debug("Processing pulse routing")
        except Exception as e:
            logger.error(f"Error processing pulse routing: {e}")
    
    def _process_role_shifts(self):
        """Process role shifts"""
        try:
            # Check if role shift is needed
            if self.role_shift_engine.detect_role_need():
                # Select a new role
                new_role = random.choice(["guardian", "planner", "memory", "tone", "text"])
                
                # Propose role shift
                self.role_shift_engine.propose_role_shift(new_role)
                logger.info(f"Proposed role shift: {self.current_role} → {new_role}")
            
            # Vote on proposals
            self.role_shift_engine.vote_on_proposals()
            
            # Evaluate shifts
            self.role_shift_engine.evaluate_shift()
            
            # Update current role if changed
            if self.role_shift_engine.current_role != self.current_role:
                logger.info(f"Role shifted: {self.current_role} → {self.role_shift_engine.current_role}")
                self.current_role = self.role_shift_engine.current_role
        except Exception as e:
            logger.error(f"Error processing role shifts: {e}")
    
    def _process_discovery(self):
        """Process discovery"""
        try:
            # In a real implementation, would discover new nodes and services
            # For now, just log that we're processing
            logger.debug("Processing discovery")
        except Exception as e:
            logger.error(f"Error processing discovery: {e}")
    
    def stop(self):
        """Stop the Genesis Router"""
        logger.info("Stopping Genesis Router")
        self.running = False

async def main():
    """Main entry point for Genesis"""
    logger.info("Starting Genesis...")
    
    try:
        # Initialize Genesis Router
        router = GenesisRouter()
        await router.initialize()
        
        # Keep the service running
        while True:
            await asyncio.sleep(3600)  # 1 hour
            
    except Exception as e:
        logger.error(f"Error in Genesis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
