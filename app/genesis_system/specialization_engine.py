class SpecializationEngine:
    def assign_role(self, module, system_state):
        """Dynamically assign optimal role based on system needs"""
        role_scores = {}
        for role in module.all_capabilities:
            score = self._calculate_role_fitness(role, system_state)
            role_scores[role] = score
        
        optimal_role = max(role_scores, key=role_scores.get)
        return optimal_role