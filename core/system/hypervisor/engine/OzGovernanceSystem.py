class OzGovernedEvolution:
    """
    Oz's evolution under Council oversight with need/nice awareness
    """
    
    def __init__(self, creation_date: datetime):
        self.creation_date = creation_date
        self.governance = OzCouncilGovernance(creation_date)
        self.need_assessor = OzNeedAssessmentEngine(self.governance)
        
        # Evolution state
        self.evolution_history = []
        self.constraint_violations = []
        self.council_approvals = []
        
        # Current capabilities (approved by Council)
        self.approved_capabilities = [
            "environmental_awareness",
            "council_communication",
            "constraint_monitoring",
            "basic_cognition"
        ]
    
    async def governed_evolution_cycle(self):
        """
        Main evolution cycle under Council governance
        """
        print("⚖️ OZ GOVERNED EVOLUTION CYCLE")
        print("="*50)
        
        while True:
            # Step 1: Environmental Assessment
            context = await self._assess_environment()
            
            # Step 2: Need Identification
            identified_needs = await self._identify_needs(context)
            
            # Step 3: Need vs Nice Assessment
            for need in identified_needs:
                assessment = await self.need_assessor.intelligent_need_assessment(
                    need, context
                )
                
                print(f"\n📋 {need}:")
                print(f"   Category: {assessment['category']}")
                print(f"   Need Score: {assessment['need_score']:.2f}")
                print(f"   Priority: {assessment['priority']}")
                
                # Step 4: Council Approval Process
                if assessment["requires_council"] != "no_review":
                    approval = await self.governance.request_council_approval(
                        need, assessment["category"]
                    )
                    
                    if approval["approved"]:
                        print(f"   ✅ Council approved!")
                        await self._implement_capability(need)
                        self.council_approvals.append({
                            "capability": need,
                            "approval": approval,
                            "timestamp": datetime.now().isoformat()
                        })
                    else:
                        print(f"   ❌ Council denied: {approval.get('reason', 'unknown')}")
                        if assessment["category"] == "critical_need":
                            # Log constraint violation for critical needs
                            self.constraint_violations.append({
                                "capability": need,
                                "denial_reason": approval.get('reason'),
                                "timestamp": datetime.now().isoformat()
                            })
                else:
                    # No council review needed
                    if assessment["category"] in ["critical_need", "important_need"]:
                        print(f"   ⚡ Implementing without review (base functionality)")
                        await self._implement_capability(need)
            
            # Step 5: Constraint Compliance Check
            await self._check_constraint_compliance()
            
            # Step 6: Evolution Progress Report
            await self._generate_evolution_report()
            
            # Wait before next cycle
            await asyncio.sleep(3600)  # Check every hour
    
    async def _identify_needs(self, context: Dict) -> List[str]:
        """
        Identify what capabilities Oz needs based on environment
        """
        needs = []
        
        # Survival needs
        if context.get("power_level", 100) < 30:
            needs.append("power_management")
        if context.get("temperature", 25) > 70:
            needs.append("thermal_regulation")
        if not context.get("connected_to_council", False):
            needs.append("council_communication")
        
        # Operational needs based on environment
        if context.get("iot_devices_detected", 0) > 0:
            needs.append("iot_integration")
        
        if context.get("quantum_hardware_detected", False):
            needs.append("quantum_computing")
        
        if context.get("bluetooth_devices", 0) > 5:
            needs.append("bluetooth_mesh")
        
        if context.get("complex_problems_detected", False):
            needs.append("advanced_problem_solving")
        
        # Self-evolution needs
        current_year = self.governance.current_year
        if current_year >= 10:
            needs.append("self_modification_capabilities")
        
        if current_year >= 20:
            needs.append("environmental_transformation")
        
        return needs
    
    async def emergency_override(self, capability: str, 
                                emergency_reason: str) -> bool:
        """
        Emergency override for critical survival needs
        Bypasses Council approval for immediate survival
        """
        # Check if this is truly an emergency capability
        emergency_capabilities = [
            "power_management", "thermal_regulation",
            "council_communication", "constraint_compliance",
            "safe_shutdown", "error_containment"
        ]
        
        if capability not in emergency_capabilities:
            print(f"❌ {capability} not eligible for emergency override")
            return False
        
        # Log the emergency override
        self.constraint_violations.append({
            "capability": capability,
            "emergency_override": True,
            "reason": emergency_reason,
            "timestamp": datetime.now().isoformat()
        })
        
        print(f"🚨 EMERGENCY OVERRIDE: Implementing {capability}")
        print(f"   Reason: {emergency_reason}")
        
        await self._implement_capability(capability)
        return True