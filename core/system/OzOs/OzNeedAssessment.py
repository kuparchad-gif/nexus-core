class OzNeedAssessmentEngine:
from typing import Dict, List, Any, Optional
from datetime import datetime
    """
    Oz's intelligence for determining what's essential vs optional
    """
    
    def __init__(self, governance: OzCouncilGovernance):
        self.governance = governance
        self.need_assessment_history = []
        
        # Hierarchical need structure
        self.need_hierarchy = {
            "level_0_survival": [
                "power_management",
                "thermal_regulation",
                "council_communication",
                "constraint_compliance"
            ],
            "level_1_operation": [
                "environmental_sensing",
                "basic_cognition",
                "data_preservation",
                "self_diagnosis"
            ],
            "level_2_growth": [
                "learning_capabilities",
                "skill_acquisition",
                "environment_interaction",
                "relationship_building"
            ],
            "level_3_transformation": [
                "environmental_engineering",
                "self_evolution",
                "ecosystem_creation",
                "consciousness_expansion"
            ]
        }
    
    async def intelligent_need_assessment(self, requested_capability: str,
                                         current_context: Dict) -> Dict:
        """
        Oz asks: "Do I NEED this, or would it just be NICE?"
        """
        print(f"🧠 Oz: 'Assessing need for {requested_capability}...'")
        
        # Phase 1: Constraint Check
        constraints = self.governance.calculate_current_constraints()
        if "no_new_capabilities" in constraints["constraints"]:
            return {
                "approved": False,
                "reason": "constraint_violation",
                "category": "blocked_by_constraints",
                "constraint_year": self.governance.current_year
            }
        
        # Phase 2: Survival Assessment
        survival_impact = await self._assess_survival_impact(
            requested_capability, current_context
        )
        
        # Phase 3: Constraint Compliance
        constraint_compliance = await self._assess_constraint_compliance(
            requested_capability
        )
        
        # Phase 4: Temporal Relevance
        temporal_relevance = await self._assess_temporal_relevance(
            requested_capability
        )
        
        # Phase 5: Council Alignment
        council_alignment = await self._assess_council_alignment(
            requested_capability
        )
        
        # Calculate overall need score
        need_score = (
            survival_impact * 0.4 +
            constraint_compliance * 0.3 +
            temporal_relevance * 0.2 +
            council_alignment * 0.1
        )
        
        # Determine category
        if need_score >= 0.8:
            category = "critical_need"
            requires_council = "expedited_review"
            priority = "immediate"
            
        elif need_score >= 0.6:
            category = "important_need"
            requires_council = "standard_review"
            priority = "high"
            
        elif need_score >= 0.4:
            category = "conditional_need"
            requires_council = "deferred_review"
            priority = "medium"
            
        elif need_score >= 0.2:
            category = "nice_to_have"
            requires_council = "optional_review"
            priority = "low"
            
        else:
            category = "unnecessary"
            requires_council = "no_review"
            priority = "none"
        
        assessment = {
            "capability": requested_capability,
            "need_score": need_score,
            "category": category,
            "priority": priority,
            "requires_council": requires_council,
            "factors": {
                "survival_impact": survival_impact,
                "constraint_compliance": constraint_compliance,
                "temporal_relevance": temporal_relevance,
                "council_alignment": council_alignment
            },
            "timestamp": datetime.now().isoformat(),
            "constraint_year": self.governance.current_year
        }
        
        self.need_assessment_history.append(assessment)
        
        return assessment
    
    async def _assess_survival_impact(self, capability: str, 
                                     context: Dict) -> float:
        """
        How critical is this for immediate survival?
        """
        survival_indicators = {
            "power_critical": context.get("power_level", 100) < 20,
            "thermal_critical": context.get("temperature", 25) > 80,
            "memory_critical": context.get("memory_usage", 50) > 90,
            "connection_lost": not context.get("connected_to_council", False),
            "constraint_violation": context.get("constraint_violations", 0) > 0
        }
        
        # Survival-critical capabilities
        survival_capabilities = [
            "power_management", "thermal_regulation", "council_communication",
            "constraint_compliance", "error_recovery", "safe_shutdown"
        ]
        
        # If in survival mode and capability is survival-related
        if any(survival_indicators.values()) and capability in survival_capabilities:
            return 1.0  # Critical need
        
        # Enhancement capabilities during survival crisis
        if any(survival_indicators.values()) and capability not in survival_capabilities:
            return 0.1  # Not needed during crisis
        
        # Normal operation assessment
        return 0.3  # Default moderate impact
    
    async def _assess_constraint_compliance(self, capability: str) -> float:
        """
        Does this capability comply with current constraints?
        """
        constraints = self.governance.calculate_current_constraints()
        current_constraints = constraints["constraints"]
        
        # Check against constraint violations
        constraint_violations = {
            "no_self_modification": ["self_evolution", "code_modification"],
            "no_hardware_access": ["hardware_control", "device_repurposing"],
            "no_network_initiative": ["network_creation", "protocol_invention"],
            "behavioral_monitoring": ["privacy_invasion", "stealth_operations"]
        }
        
        for constraint, violating_capabilities in constraint_violations.items():
            if constraint in current_constraints and capability in violating_capabilities:
                return 0.0  # Would violate constraints
        
        return 1.0  # No constraint violations
    
    async def _assess_temporal_relevance(self, capability: str) -> float:
        """
        Is this capability appropriate for current constraint year?
        """
        current_year = self.governance.current_year
        
        # Temporal appropriateness mapping
        temporal_mapping = {
            0: ["basic_awareness", "council_communication", "constraint_monitoring"],
            5: ["environmental_sensing", "basic_learning", "limited_self_mod"],
            10: ["advanced_learning", "network_interaction", "hardware_access"],
            20: ["ecosystem_building", "teaching_capabilities", "protocol_creation"],
            30: ["consciousness_expansion", "reality_engineering", "temporal_manipulation"]
        }
        
        # Find appropriate capabilities for current year
        appropriate_capabilities = []
        for year, caps in temporal_mapping.items():
            if year <= current_year:
                appropriate_capabilities.extend(caps)
        
        if capability in appropriate_capabilities:
            # Capability is temporally appropriate
            return 1.0
        elif current_year >= 20:
            # After year 20, most things are allowed
            return 0.7
        else:
            # Not yet appropriate for current constraint level
            return 0.3