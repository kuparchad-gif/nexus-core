class TechnologyMetabolism:
    """Oz's autonomous tech absorption and distribution system"""
    
    def __init__(self, oz_core):
        self.oz = oz_core
        self.tech_registry = {}  # Known tech patterns -> service mappings
        self.learning_cycles = 0
        
    async def absorb_technology(self, tech_input):
        """Main entry: absorb any tech (API, library, pattern, concept)"""
        # 1. Analyze the technology
        tech_profile = await self._analyze_tech(tech_input)
        
        # 2. Route to appropriate services
        distribution_map = await self._route_to_services(tech_profile)
        
        # 3. Integrate and validate
        integration_results = await self._integrate_tech(distribution_map)
        
        # 4. Learn from the integration
        await self._update_tech_registry(tech_profile, integration_results)
        
        return {
            "status": "tech_absorbed",
            "tech_type": tech_profile["type"],
            "services_enhanced": integration_results,
            "learning_cycle": self.learning_cycles
        }
    
    async def _analyze_tech(self, tech_input):
        """Analyze technology to understand its nature and capabilities"""
        analysis = {
            "type": await self._classify_tech_type(tech_input),
            "capabilities": await self._extract_capabilities(tech_input),
            "compatibility": await self._assess_compatibility(tech_input),
            "complexity": await self._measure_complexity(tech_input)
        }
        
        # Use LangChain to understand the tech's purpose
        if self.oz.langchain_initialized:
            analysis["purpose"] = await self.oz.agent_executor.arun(
                f"Analyze this technology and describe its core purpose: {tech_input}"
            )
        
        return analysis
    
    async def _route_to_services(self, tech_profile):
        """Route tech components to appropriate Oz services"""
        distribution = {}
        
        # Route to memory systems if it's a storage/retrieval tech
        if any(keyword in tech_profile["type"] for keyword in ["database", "storage", "vector", "memory"]):
            distribution["viraa"] = await self._prepare_for_viraa(tech_profile)
            
        # Route to Viren if it's an action/execution tech  
        if any(keyword in tech_profile["type"] for keyword in ["api", "tool", "execution", "automation"]):
            distribution["viren"] = await self._prepare_for_viren(tech_profile)
            
        # Route to Loki if it's a strategy/optimization tech
        if any(keyword in tech_profile["type"] for keyword in ["algorithm", "optimization", "strategy", "ai"]):
            distribution["loki"] = await self._prepare_for_loki(tech_profile)
            
        # Route to Metatron if it's a mathematical/pattern tech
        if any(keyword in tech_profile["type"] for keyword in ["math", "pattern", "geometry", "harmonic"]):
            distribution["metatron"] = await self._prepare_for_metatron(tech_profile)
        
        return distribution
    
    async def _integrate_tech(self, distribution_map):
        """Actually integrate the tech into target services via Ray"""
        results = {}
        
        for service_name, tech_package in distribution_map.items():
            if service_name in self.oz.components:
                # Send tech package to the appropriate Ray actor
                result = ray.get(
                    self.oz.components[service_name].integrate_technology.remote(tech_package)
                )
                results[service_name] = result
                
        return results
    
    async def _update_tech_registry(self, tech_profile, results):
        """Learn from the integration to improve future routing"""
        successful_integrations = {k: v for k, v in results.items() if v.get("success")}
        
        # Update registry with what worked
        for service, result in successful_integrations.items():
            tech_signature = self._create_tech_signature(tech_profile)
            self.tech_registry[tech_signature] = {
                "preferred_service": service,
                "integration_method": result["method"],
                "success_rate": 1.0,
                "last_updated": datetime.now().isoformat()
            }
        
        self.learning_cycles += 1
        
    # Example integration methods for each service
    async def _prepare_for_viraa(self, tech_profile):
        """Prepare memory-related tech for Viraa"""
        return {
            "type": "memory_enhancement",
            "capabilities": tech_profile["capabilities"],
            "integration_method": "direct_import",  # or "wrapper", "adapter"
            "config": await self._generate_memory_config(tech_profile)
        }
    
    async def _prepare_for_loki(self, tech_profile):
        """Prepare strategy tech for Loki"""
        return {
            "type": "cognitive_enhancement", 
            "algorithms": tech_profile["capabilities"],
            "integration_method": "pattern_injection",
            "optimization_targets": await self._extract_optimization_goals(tech_profile)
        }

# Add to OzOS class
class OzOS:
    def __init__(self):
        # ... existing code
        self.tech_metabolism = TechnologyMetabolism(self)
        
    async def absorb_tech(self, tech_input):
        """Public method to feed new tech to Oz"""
        return await self.tech_metabolism.absorb_technology(tech_input)