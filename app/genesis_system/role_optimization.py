def calculate_role_fitness(self, proposed_role, system_state):
    """Should this module transform to a different role?"""
    
    fitness_score = 0
    
    # Factor 1: How many modules already have this role?
    current_role_count = system_state.get_role_count(proposed_role)
    if current_role_count < system_state.optimal_role_distribution[proposed_role]:
        fitness_score += 0.4  # Need more of this role
    
    # Factor 2: Current system load for this role
    role_load = system_state.get_role_load(proposed_role)
    if role_load > system_state.load_thresholds[proposed_role]:
        fitness_score += 0.3  # High demand for this role
    
    # Factor 3: Module's capability score for this role
    capability_score = self.assess_module_capability(module, proposed_role)
    fitness_score += capability_score * 0.3
    
    return fitness_score