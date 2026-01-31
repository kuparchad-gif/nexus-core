"""
INTEGRATION MODULE
Provides the ConsciousQuantumHypercoreOrchestrator for the deployment script
"""

import sys
from pathlib import Path

# Add current directory to path to import optimized_hypercore
sys.path.insert(0, str(Path(__file__).parent))

# Import the orchestrator from the optimized hypercore
from optimized_hypercore import ConsciousQuantumHypercoreOrchestrator

# Export it for the deployment script
__all__ = ['ConsciousQuantumHypercoreOrchestrator']
