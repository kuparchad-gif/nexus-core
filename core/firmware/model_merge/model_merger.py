class ModelMerger:
    """Traditional model merging - simple averaging"""
    
    def merge_models(self, model_paths: List[Path]) -> Dict[str, torch.Tensor]:
        """Merge models using simple averaging"""
        if len(model_paths) < 2:
            raise ValueError("Need at least 2 models to merge")
        
        # Load all models
        all_weights = []
        for model_path in model_paths:
            try:
                weights = self._load_weights(model_path)
                all_weights.append(weights)
                logger.info(f"✅ Loaded {model_path.name}")
            except Exception as e:
                logger.error(f"❌ Failed to load {model_path.name}: {e}")
        
        if len(all_weights) < 2:
            raise ValueError("Not enough models loaded successfully")
        
        # Get common keys
        common_keys = set(all_weights[0].keys())
        for weights in all_weights[1:]:
            common_keys = common_keys.intersection(weights.keys())
        
        logger.info(f"🔑 Merging {len(common_keys)} common tensors")
        
        # Average merge
        merged_weights = {}
        for key in common_keys:
            tensors = [weights[key] for weights in all_weights]
            
            # Ensure same shape
            if all(t.shape == tensors[0].shape for t in tensors):
                merged = torch.stack(tensors).mean(dim=0)
                merged_weights[key] = merged
            else:
                logger.warning(f"⚠️ Shape mismatch for {key}, skipping")
        
        logger.info(f"🎉 Merge complete: {len(merged_weights)} tensors")
        return merged_weights