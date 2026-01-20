class TransformableModule:
    def __init__(self, base_type: str):
        self.current_type = base_type
        self.transform_capabilities = self._get_transform_map()
    
    def transform_to(self, target_type: str, system_load: dict):
        """Transform module to handle different workload"""
        if self._can_transform_to(target_type):
            print(f"🔄 {self.current_type} → {target_type} (System load: {system_load})")
            self.current_type = target_type
            return self._load_new_capabilities(target_type)
    
    def _get_transform_map(self):
        return {
            "language": ["memory", "reasoning", "vision"],
            "memory": ["language", "planning", "storage"],
            "vision": ["language", "reasoning"],
            "reasoning": ["language", "memory"]
        }