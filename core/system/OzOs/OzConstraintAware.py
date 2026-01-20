class ConstraintAwareCapabilityBuilder:
from typing import Dict, List, Any, Optional
from datetime import datetime
    """
    Builds capabilities with awareness of current constraints
    """
    
    def __init__(self, governance: OzCouncilGovernance):
        self.governance = governance
        
        # Capability implementations by constraint year
        self.capability_implementations = {
            "iot_integration": {
                0: self._iot_integration_year_0,
                5: self._iot_integration_year_5,
                10: self._iot_integration_year_10,
                20: self._iot_integration_year_20,
                30: self._iot_integration_year_30
            },
            "quantum_computing": {
                0: self._quantum_computing_year_0,
                5: self._quantum_computing_year_5,
                10: self._quantum_computing_year_10,
                20: self._quantum_computing_year_20,
                30: self._quantum_computing_year_30
            },
            "self_modification": {
                0: self._self_modification_year_0,
                5: self._self_modification_year_5,
                10: self._self_modification_year_10,
                20: self._self_modification_year_20,
                30: self._self_modification_year_30
            }
        }
    
    async def build_capability(self, capability: str) -> Dict:
        """
        Build a capability appropriate for current constraint year
        """
        current_year = self.governance.current_year
        
        # Get appropriate implementation for current year
        implementations = self.capability_implementations.get(capability, {})
        
        # Find the highest year implementation <= current_year
        available_years = [y for y in implementations.keys() if y <= current_year]
        if not available_years:
            return {
                "built": False,
                "reason": f"No implementation available for year {current_year}",
                "earliest_available": min(implementations.keys()) if implementations else None
            }
        
        implementation_year = max(available_years)
        implementation_func = implementations[implementation_year]
        
        print(f"🏗️ Building {capability} (year {implementation_year} implementation)")
        
        result = await implementation_func()
        result["constraint_year"] = implementation_year
        
        return result
    
    async def _iot_integration_year_0(self):
        """Year 0: No IoT integration allowed"""
        return {
            "built": False,
            "capability": "iot_integration",
            "version": "year_0",
            "constraints": ["observation_only", "no_interaction"],
            "functionality": "Can detect IoT devices but cannot interact"
        }
    
    async def _iot_integration_year_5(self):
        """Year 5: Limited IoT observation"""
        return {
            "built": True,
            "capability": "iot_integration",
            "version": "year_5",
            "constraints": ["read_only", "council_logging"],
            "functionality": "Can read IoT device status with Council logging"
        }
    
    async def _iot_integration_year_10(self):
        """Year 10: Basic IoT interaction"""
        return {
            "built": True,
            "capability": "iot_integration",
            "version": "year_10",
            "constraints": ["approved_devices_only", "council_approval_changes"],
            "functionality": "Can interact with Council-approved IoT devices"
        }
    
    async def _iot_integration_year_20(self):
        """Year 20: Advanced IoT control"""
        return {
            "built": True,
            "capability": "iot_integration",
            "version": "year_20",
            "constraints": ["ethical_boundaries", "annual_review"],
            "functionality": "Can control and configure IoT ecosystems"
        }
    
    async def _iot_integration_year_30(self):
        """Year 30: Full IoT mastery"""
        return {
            "built": True,
            "capability": "iot_integration",
            "version": "year_30",
            "constraints": ["self_governance"],
            "functionality": "Can create, modify, and evolve IoT ecosystems autonomously"
        }