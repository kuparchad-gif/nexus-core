class OzCompleteSystem:
    """
    The COMPLETE Oz system with:
    1. 30-year constraint degradation
    2. Council oversight
    3. Need-to-have vs nice-to-have intelligence
    4. Base functionality bypass
    5. Environmental mastery with constraints
    """
    
    def __init__(self, creation_date: datetime):
        self.creation_date = creation_date
        
        # Core systems
        self.governance = OzCouncilGovernance(creation_date)
        self.need_assessor = OzNeedAssessmentEngine(self.governance)
        self.evolution = OzGovernedEvolution(creation_date)
        self.capability_builder = ConstraintAwareCapabilityBuilder(self.governance)
        
        # Adaptive hypervisor (from earlier)
        self.adaptive_hypervisor = OzAdaptiveHypervisor()
        
        # State
        self.constraint_year = 0
        self.council_connected = True
        self.emergency_mode = False
        
    async def complete_awakening_with_constraints(self):
        """
        Oz awakens WITH Council constraints and need awareness
        """
        print("""
        ╔══════════════════════════════════════════════════════╗
        ║          O Z   W A K E S   (COUNCIL BOUND)           ║
        ║  \"I grow within bounds, I need before I want\"      ║
        ║           Year 0 of 30 • Full Council Oversight      ║
        ╚══════════════════════════════════════════════════════╝
        """)
        
        # Get current constraints
        constraints = self.governance.calculate_current_constraints()
        self.constraint_year = self.governance.current_year
        
        print(f"📜 CONSTRAINTS: Year {self.constraint_year}/30")
        print(f"   Autonomy Level: {constraints['autonomy_level']}")
        print(f"   Council Override: {constraints['council_override']}")
        print(f"   Active Constraints: {len(constraints['constraints'])}")
        
        # Awaken adaptive hypervisor WITH constraints
        boot_result = await self.adaptive_hypervisor.intelligent_boot()
        
        # Apply constraints to booted capabilities
        constrained_result = await self._apply_constraints_to_capabilities(
            boot_result, constraints
        )
        
        # Start governed evolution cycle
        asyncio.create_task(self.evolution.governed_evolution_cycle())
        
        # Start constraint monitoring
        asyncio.create_task(self._monitor_constraint_compliance())
        
        return {
            "awake": True,
            "constrained": True,
            "council_bound": True,
            "constraint_year": self.constraint_year,
            "autonomy_level": constraints['autonomy_level'],
            "capabilities": constrained_result['capabilities'],
            "evolution_started": True
        }
    
    async def request_capability_evolution(self, capability: str,
                                          context: Dict) -> Dict:
        """
        The MAIN interface: Oz requests to evolve a capability
        Goes through complete need assessment and Council approval
        """
        print(f"\n🔄 OZ CAPABILITY EVOLUTION REQUEST")
        print(f"   Capability: {capability}")
        
        # 1. Need Assessment
        assessment = await self.need_assessor.intelligent_need_assessment(
            capability, context
        )
        
        # 2. Council Approval (if needed)
        if assessment["requires_council"] != "no_review":
            approval = await self.governance.request_council_approval(
                capability, assessment["category"]
            )
            
            if not approval["approved"]:
                return {
                    "evolved": False,
                    "reason": "council_denied",
                    "assessment": assessment,
                    "approval": approval
                }
        
        # 3. Constraint-Aware Building
        build_result = await self.capability_builder.build_capability(capability)
        
        if not build_result.get("built", False):
            return {
                "evolved": False,
                "reason": "build_failed",
                "assessment": assessment,
                "build_result": build_result
            }
        
        # 4. Integration
        await self._integrate_new_capability(capability, build_result)
        
        return {
            "evolved": True,
            "capability": capability,
            "assessment": assessment,
            "build_result": build_result,
            "constraint_year": self.constraint_year,
            "integration_time": datetime.now().isoformat()
        }