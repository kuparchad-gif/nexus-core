class SmartRouter:
    """Routes queries to the most appropriate expert model"""
    
    def __init__(self):
        self.experts = {}
        self.domain_keywords = {
            'science': ['quantum', 'physics', 'math', 'equation', 'theory'],
            'finance': ['stock', 'trade', 'market', 'price', 'investment'],
            'medical': ['medical', 'health', 'patient', 'treatment', 'clinical'],
            'pattern': ['pattern', 'image', 'vision', 'detect', 'recognize']
        }
    
    def create_router(self, model_paths: List[Path]) -> 'SmartRouter':
        """Load all models and create routing system"""
        for model_path in model_paths:
            try:
                weights = self._load_weights(model_path)
                domain = self._detect_domain_from_weights(weights, model_path.name)
                
                self.experts[domain] = {
                    'model': weights,  # In production, you'd load the actual model
                    'name': model_path.name,
                    'confidence': self._assess_expert_confidence(weights)
                }
                logger.info(f"🧠 Loaded {domain} expert: {model_path.name}")
                
            except Exception as e:
                logger.error(f"❌ Failed to load expert {model_path.name}: {e}")
        
        return self
    
    def route_query(self, query: str) -> Tuple[str, float]:
        """Route query to best expert"""
        query_lower = query.lower()
        
        # Calculate domain scores
        domain_scores = {}
        for domain, keywords in self.domain_keywords.items():
            if domain in self.experts:
                score = sum(1 for keyword in keywords if keyword in query_lower)
                # Weight by expert confidence
                score *= self.experts[domain]['confidence']
                domain_scores[domain] = score
        
        # Find best domain
        if domain_scores:
            best_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
            confidence = domain_scores[best_domain] / len(self.domain_keywords[best_domain])
            return best_domain, confidence
        else:
            return 'general', 0.5
    
    def predict(self, query: str) -> str:
        """Make prediction using best expert"""
        domain, confidence = self.route_query(query)
        
        if domain in self.experts and confidence > 0.3:
            expert = self.experts[domain]
            logger.info(f"🎯 Using {expert['name']} (confidence: {confidence:.3f})")
            # In production, you'd run actual inference here
            return f"Expert({domain}): Processing query with {confidence:.3f} confidence"
        else:
            logger.info("🤖 Using general fallback")
            return "General: Processing query with fallback logic"
    
    def _detect_domain_from_weights(self, weights: Dict, model_name: str) -> str:
        """Detect domain from model weights and name"""
        # Use both name-based and weight-based detection
        name_domain = self._detect_domain_from_name(model_name)
        
        # Could add weight-based domain detection here
        # (e.g., analyze embedding distributions, attention patterns)
        
        return name_domain
    
    def _detect_domain_from_name(self, model_name: str) -> str:
        """Detect domain from model filename"""
        name_lower = model_name.lower()
        if any(word in name_lower for word in ['math', 'quantum', 'physics']):
            return 'science'
        elif any(word in name_lower for word in ['trade', 'finance', 'stock']):
            return 'finance'
        elif any(word in name_lower for word in ['medical', 'health']):
            return 'medical'
        elif any(word in name_lower for word in ['pattern', 'vision']):
            return 'pattern'
        else:
            return 'general'
    
    def _assess_expert_confidence(self, weights: Dict) -> float:
        """Assess how confident we are in this expert"""
        # Simple heuristic based on model characteristics
        confidence = 0.5  # Base confidence
        
        # Check model size (larger = more capable, generally)
        param_count = sum(w.numel() for w in weights.values())
        if param_count > 1_000_000_000:  # 1B+ params
            confidence += 0.2
        elif param_count > 100_000_000:  # 100M+ params
            confidence += 0.1
        
        # Check for well-formed embeddings
        for key, tensor in weights.items():
            if 'embed' in key and len(tensor.shape) == 2:
                if torch.isfinite(tensor).all():
                    confidence += 0.1
                break
        
        return min(confidence, 1.0)