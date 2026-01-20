class ArchitecturalDigestor:
    """Absorbs intelligence while filtering stupidity"""
    
    def __init__(self):
        self.base_model = None
        self.absorption_stats = {
            'parameters_absorbed': 0,
            'stupidity_filtered': 0,
            'models_digested': 0
        }
    
    def digest_models(self, model_paths: List[Path]) -> Dict[str, torch.Tensor]:
        """Digest multiple models intelligently"""
        if not model_paths:
            raise ValueError("No models to digest")
        
        # Start with highest quality model as base
        self.base_model = self._select_base_model(model_paths)
        logger.info(f"🧠 Selected base model: {self.base_model['name']}")
        
        # Digest other models
        for model_path in model_paths:
            if model_path.name != self.base_model['name']:
                self._digest_single_model(model_path)
        
        efficiency = self.absorption_stats['parameters_absorbed'] / (
            self.absorption_stats['parameters_absorbed'] + 
            self.absorption_stats['stupidity_filtered'] + 1e-8
        )
        
        logger.info(f"📊 Digestion complete: {efficiency:.3f} efficiency")
        return self.base_model['weights']
    
    def _select_base_model(self, model_paths: List[Path]) -> Dict[str, Any]:
        """Select the best model as base for digestion"""
        best_model = None
        best_score = -1
        
        for model_path in model_paths:
            try:
                weights = self._load_weights(model_path)
                score = self._score_model_quality(weights)
                
                if score > best_score:
                    best_score = score
                    best_model = {
                        'name': model_path.name,
                        'weights': weights,
                        'score': score
                    }
            except Exception as e:
                logger.error(f"❌ Failed to score {model_path.name}: {e}")
        
        return best_model
    
    def _digest_single_model(self, model_path: Path):
        """Digest a single donor model"""
        try:
            donor_weights = self._load_weights(model_path)
            logger.info(f"🍽️ Digesting {model_path.name}...")
            
            absorbed_params = 0
            filtered_params = 0
            
            for key in self.base_model['weights']:
                if key in donor_weights:
                    base_tensor = self.base_model['weights'][key]
                    donor_tensor = donor_weights[key]
                    
                    # Check if absorption makes sense
                    if self._should_absorb(key, base_tensor, donor_tensor):
                        # Intelligent absorption
                        absorbed = self._absorb_intelligently(base_tensor, donor_tensor)
                        self.base_model['weights'][key] = absorbed
                        absorbed_params += base_tensor.numel()
                    else:
                        filtered_params += base_tensor.numel()
            
            self.absorption_stats['parameters_absorbed'] += absorbed_params
            self.absorption_stats['stupidity_filtered'] += filtered_params
            self.absorption_stats['models_digested'] += 1
            
            logger.info(f"   Absorbed: {absorbed_params:,}, Filtered: {filtered_params:,}")
            
        except Exception as e:
            logger.error(f"❌ Failed to digest {model_path.name}: {e}")
    
    def _should_absorb(self, key: str, base: torch.Tensor, donor: torch.Tensor) -> bool:
        """Check if absorption makes sense for this tensor"""
        # Shape compatibility
        if base.shape != donor.shape:
            return False
        
        # Semantic compatibility for embeddings
        if 'embed' in key:
            return self._check_semantic_compatibility(base, donor)
        
        # General tensor compatibility
        base_norm = torch.norm(base)
        donor_norm = torch.norm(donor)
        norm_ratio = donor_norm / (base_norm + 1e-8)
        
        return 0.1 < norm_ratio < 10.0  # Within reasonable range
    
    def _check_semantic_compatibility(self, base: torch.Tensor, donor: torch.Tensor) -> bool:
        """Check if embedding spaces are semantically compatible"""
        # Compare centroids
        base_centroid = torch.mean(base, dim=0)
        donor_centroid = torch.mean(donor, dim=0)
        
        similarity = torch.cosine_similarity(base_centroid, donor_centroid, dim=0)
        return similarity > 0.3  # Some semantic overlap
    
    def _absorb_intelligently(self, base: torch.Tensor, donor: torch.Tensor) -> torch.Tensor:
        """Intelligently absorb donor intelligence into base"""
        absorption_strength = 0.2  # Conservative absorption
        
        # Weighted absorption based on tensor characteristics
        base_quality = self._assess_tensor_quality(base)
        donor_quality = self._assess_tensor_quality(donor)
        
        # Absorb more from higher quality donors
        effective_strength = absorption_strength * (donor_quality / base_quality)
        effective_strength = min(effective_strength, 0.5)  # Cap absorption
        
        return (1 - effective_strength) * base + effective_strength * donor
    
    def _score_model_quality(self, weights: Dict) -> float:
        """Score model quality (0-1)"""
        quality_indicators = 0
        total_indicators = 0
        
        for tensor in weights.values():
            # Check for finite values
            if torch.isfinite(tensor).all():
                quality_indicators += 1
            
            # Check for reasonable norms
            norm = torch.norm(tensor).item()
            if 0.001 < norm < 1000.0:
                quality_indicators += 1
            
            total_indicators += 2
        
        return quality_indicators / max(total_indicators, 1)
    
    def _assess_tensor_quality(self, tensor: torch.Tensor) -> float:
        """Assess quality of individual tensor"""
        quality = 0.5  # Base quality
        
        # Check for finite values
        if torch.isfinite(tensor).all():
            quality += 0.2
        
        # Check for reasonable scale
        norm = torch.norm(tensor).item()
        if 0.01 < norm < 100.0:
            quality += 0.3
        
        return min(quality, 1.0)