#!/usr/bin/env python3
"""
METATRON ORCHESTRATOR
Chooses the right merging strategy based on model characteristics
"""

import torch
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging
from safetensors.torch import load_file
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MetatronOrchestrator:
    """Intelligently chooses between merging strategies"""
    
    def __init__(self):
        self.strategies = {
            'smart_routing': SmartRouter(),
            'absorption': ArchitecturalDigestor(), 
            'merging': ModelMerger(),
            'adapter_switching': AdapterSwitcher()
        }
        
    def analyze_models(self, model_paths: List[Path]) -> Dict[str, Any]:
        """Analyze model compatibility and characteristics"""
        analysis = {
            'architectures': set(),
            'vocab_sizes': set(),
            'parameter_counts': [],
            'quality_scores': {},
            'domains': {}
        }
        
        for model_path in model_paths:
            try:
                weights = self._load_weights(model_path)
                
                # Architecture analysis
                arch = self._detect_architecture(weights)
                analysis['architectures'].add(arch)
                
                # Vocabulary analysis
                vocab_size = self._detect_vocabulary_size(weights)
                analysis['vocab_sizes'].add(vocab_size)
                
                # Parameter count
                param_count = sum(w.numel() for w in weights.values())
                analysis['parameter_counts'].append(param_count)
                
                # Quality assessment (heuristic)
                quality = self._assess_model_quality(weights)
                analysis['quality_scores'][model_path.name] = quality
                
                # Domain detection
                domain = self._detect_domain(model_path.name)
                analysis['domains'][model_path.name] = domain
                
            except Exception as e:
                logger.error(f"❌ Failed to analyze {model_path.name}: {e}")
        
        return analysis
    
    def recommend_strategy(self, analysis: Dict[str, Any], use_case: str) -> str:
        """Recommend the best strategy based on analysis"""
        
        # Strategy decision tree
        architectures = analysis['architectures']
        vocab_sizes = analysis['vocab_sizes']
        domains = set(analysis['domains'].values())
        avg_quality = np.mean(list(analysis['quality_scores'].values()))
        
        logger.info(f"📊 Analysis: {len(architectures)} arch, {len(vocab_sizes)} vocabs, {len(domains)} domains, quality: {avg_quality:.3f}")
        
        # Decision logic
        if len(architectures) > 3 or len(vocab_sizes) > 2:
            logger.info("🎯 Recommendation: SMART ROUTING (too diverse)")
            return 'smart_routing'
            
        elif len(domains) > 2 and use_case == "production":
            logger.info("🎯 Recommendation: ADAPTER SWITCHING (multi-domain production)")
            return 'adapter_switching'
            
        elif len(architectures) == 1 and len(vocab_sizes) == 1:
            if avg_quality > 0.7:
                logger.info("🎯 Recommendation: MODEL MERGING (compatible & high quality)")
                return 'merging'
            else:
                logger.info("🎯 Recommendation: ABSORPTION (compatible but mixed quality)")
                return 'absorption'
                
        elif len(architectures) == 1 and avg_quality > 0.6:
            logger.info("🎯 Recommendation: ABSORPTION (same arch, decent quality)")
            return 'absorption'
            
        else:
            logger.info("🎯 Recommendation: SMART ROUTING (fallback - too complex)")
            return 'smart_routing'
    
    def execute_strategy(self, model_paths: List[Path], strategy: str, use_case: str) -> Any:
        """Execute the chosen strategy"""
        logger.info(f"🚀 Executing strategy: {strategy}")
        
        if strategy == 'smart_routing':
            return self.strategies['smart_routing'].create_router(model_paths)
        elif strategy == 'absorption':
            return self.strategies['absorption'].digest_models(model_paths)
        elif strategy == 'merging':
            return self.strategies['merging'].merge_models(model_paths)
        elif strategy == 'adapter_switching':
            return self.strategies['adapter_switching'].setup_switching(model_paths)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _load_weights(self, model_path: Path) -> Dict[str, torch.Tensor]:
        """Load model weights"""
        if model_path.suffix == '.safetensors':
            return load_file(str(model_path))
        else:
            return torch.load(str(model_path), map_location='cpu')
    
    def _detect_architecture(self, weights: Dict) -> str:
        """Detect model architecture"""
        keys = list(weights.keys())
        if any('llama' in key.lower() for key in keys):
            return 'llama'
        elif any('mistral' in key.lower() for key in keys):
            return 'mistral'
        elif any('bert' in key.lower() for key in keys):
            return 'bert'
        else:
            return 'unknown'
    
    def _detect_vocabulary_size(self, weights: Dict) -> int:
        """Detect vocabulary size from embedding layers"""
        for key, tensor in weights.items():
            if 'embed' in key and len(tensor.shape) == 2:
                return tensor.shape[0]
        return 0
    
    def _assess_model_quality(self, weights: Dict) -> float:
        """Heuristic model quality assessment"""
        quality_signals = 0
        total_checks = 0
        
        # Check embedding norms (shouldn't be extreme)
        for key, tensor in weights.items():
            if 'embed' in key and len(tensor.shape) == 2:
                norm = torch.norm(tensor).item()
                if 0.1 < norm < 10.0:  # Reasonable range
                    quality_signals += 1
                total_checks += 1
        
        # Check for NaN/Inf
        for tensor in weights.values():
            if torch.isfinite(tensor).all():
                quality_signals += 1
            total_checks += 1
        
        return quality_signals / max(total_checks, 1)
    
    def _detect_domain(self, model_name: str) -> str:
        """Detect model domain from filename"""
        name_lower = model_name.lower()
        if any(word in name_lower for word in ['math', 'quantum', 'physics']):
            return 'science'
        elif any(word in name_lower for word in ['trade', 'finance', 'stock']):
            return 'finance'
        elif any(word in name_lower for word in ['medical', 'health', 'clinical']):
            return 'medical'
        elif any(word in name_lower for word in ['pattern', 'recognition', 'vision']):
            return 'pattern'
        else:
            return 'general'