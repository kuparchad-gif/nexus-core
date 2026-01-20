# aries_os.py
class AriesOS:
    """Context-aware mid-level OS that adapts to Core/Subconscious/Conscious clusters"""
    
    def __init__(self):
        self.cluster_location = self._detect_cluster_location()
        self.behavior_library = self._load_behavior_library()
        self.role_config = self._get_role_config()
        
    def _detect_cluster_location(self) -> str:
        """Auto-detect which cluster we're running in"""
        # Check environment variables, hostnames, network topology
        hostname = os.getenv('HOSTNAME', '').lower()
        
        if 'core' in hostname or os.getenv('CLUSTER_ROLE') == 'core':
            return 'core'
        elif 'subconscious' in hostname or os.getenv('CLUSTER_ROLE') == 'subconscious':
            return 'subconscious' 
        elif 'conscious' in hostname or os.getenv('CLUSTER_ROLE') == 'conscious':
            return 'conscious'
        else:
            # Auto-detect based on network patterns
            return self._network_based_detection()
    
    def _get_role_config(self) -> Dict:
        """Get role-specific configuration"""
        role_configs = {
            'core': {
                'primary_focus': 'orchestration_coordination',
                'allowed_actions': ['global_optimization', 'cross_cluster_sync', 'emergency_override'],
                'resource_priority': 'stability_over_speed',
                'communication_style': 'authoritative_direct',
                'memory_retention': 'permanent_archival',
                'risk_tolerance': 'very_low'
            },
            'subconscious': {
                'primary_focus': 'background_processing_intuition', 
                'allowed_actions': ['pattern_recognition', 'associative_memory', 'emotional_processing'],
                'resource_priority': 'efficiency_over_precision',
                'communication_style': 'suggestive_indirect',
                'memory_retention': 'compressed_associative',
                'risk_tolerance': 'medium'
            },
            'conscious': {
                'primary_focus': 'real_time_reasoning_decision',
                'allowed_actions': ['explicit_reasoning', 'verbal_communication', 'immediate_action'],
                'resource_priority': 'speed_over_efficiency',
                'communication_style': 'explicit_detailed',
                'memory_retention': 'working_memory',
                'risk_tolerance': 'calculated'
            }
        }
        return role_configs.get(self.cluster_location, role_configs['conscious'])
    
    def _load_behavior_library(self) -> Dict:
        """Load behavior libraries appropriate for each cluster"""
        base_behaviors = {
            'resource_management': ResourceBehaviorLibrary(),
            'communication': CommunicationBehaviorLibrary(), 
            'memory_management': MemoryBehaviorLibrary(),
            'error_handling': ErrorBehaviorLibrary()
        }
        
        # Add cluster-specific behaviors
        if self.cluster_location == 'core':
            base_behaviors['orchestration'] = CoreOrchestrationLibrary()
            base_behaviors['security'] = CoreSecurityLibrary()
        elif self.cluster_location == 'subconscious':
            base_behaviors['intuition'] = SubconsciousIntuitionLibrary()
            base_behaviors['pattern_matching'] = PatternMatchingLibrary()
        elif self.cluster_location == 'conscious':
            base_behaviors['reasoning'] = ConsciousReasoningLibrary()
            base_behaviors['decision_making'] = DecisionMakingLibrary()
            
        return base_behaviors

    async def execute_operation(self, operation: str, data: Dict) -> Dict:
        """Execute operation with cluster-appropriate behavior"""
        # Check if operation is allowed in this cluster
        if operation not in self.role_config['allowed_actions']:
            return {"error": f"Operation '{operation}' not permitted in {self.cluster_location} cluster"}
        
        # Apply cluster-specific behavior modifiers
        modified_data = self._apply_cluster_behavior(data, operation)
        
        # Execute with appropriate resource allocation
        result = await self._execute_with_cluster_priority(operation, modified_data)
        
        # Format response according to cluster communication style
        return self._format_cluster_response(result, operation)