class AdapterSwitcher:
    """Switches between LoRA adapters at runtime"""
    
    def __init__(self):
        self.base_model = None
        self.adapters = {}
    
    def setup_switching(self, model_paths: List[Path]) -> 'AdapterSwitcher':
        """Setup adapter switching system"""
        if not model_paths:
            raise ValueError("No models provided")
        
        # Assume first model is base, others are adapters
        self.base_model = self._load_weights(model_paths[0])
        logger.info(f"🧠 Base model: {model_paths[0].name}")
        
        # Load adapters
        for model_path in model_paths[1:]:
            domain = self._detect_domain(model_path.name)
            self.adapters[domain] = self._load_weights(model_path)
            logger.info(f"🔌 Loaded {domain} adapter: {model_path.name}")
        
        return self
    
    def switch_to_adapter(self, domain: str) -> Dict[str, torch.Tensor]:
        """Switch to specific domain adapter"""
        if domain not in self.adapters:
            available = list(self.adapters.keys())
            raise ValueError(f"Adapter {domain} not found. Available: {available}")
        
        # In production, you'd apply adapter weights to base model
        logger.info(f"🔀 Switched to {domain} adapter")
        return self.adapters[domain]
    
    def get_available_adapters(self) -> List[str]:
        """Get list of available adapters"""
        return list(self.adapters.keys())