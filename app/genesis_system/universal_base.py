class UniversalModule:
    def __init__(self):
        self.all_capabilities = ["language", "memory", "reasoning", "vision", "web", "storage"]
        self.current_role = None
        self.performance_metrics = {}