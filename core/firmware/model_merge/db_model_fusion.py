#!/usr/bin/env python3
"""
WEIGHT DATABASE FUSION SYSTEM
Store all model weights in a database, merge via queries
"""

import sqlite3
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WeightDatabase:
    """Database-driven model weight storage and fusion"""
    
    def __init__(self, db_path: Path = Path("model_weights.db")):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize the weight database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Models table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS models (
                model_id TEXT PRIMARY KEY,
                model_name TEXT,
                architecture TEXT,
                vocab_size INTEGER,
                parameter_count INTEGER,
                quality_score REAL,
                domain TEXT,
                metadata JSON
            )
        ''')
        
        # Weights table - stores individual tensors
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS weights (
                weight_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT,
                tensor_key TEXT,
                tensor_shape TEXT,  # JSON string of shape
                tensor_dtype TEXT,
                tensor_data BLOB,   # Serialized tensor data
                importance_score REAL,
                fusion_metadata JSON,
                FOREIGN KEY (model_id) REFERENCES models (model_id)
            )
        ''')
        
        # Fusion recipes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fusion_recipes (
                recipe_id INTEGER PRIMARY KEY AUTOINCREMENT,
                recipe_name TEXT,
                fusion_strategy TEXT,
                model_sources JSON,
                weight_mapping JSON,
                performance_metrics JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("🗄️ Weight database initialized")
    
    def import_model(self, model_path: Path, model_id: str) -> str:
        """Import a model into the weight database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Load model weights
        if model_path.suffix == '.safetensors':
            from safetensors.torch import load_file
            weights = load_file(str(model_path))
        else:
            weights = torch.load(str(model_path), map_location='cpu')
        
        # Analyze model
        model_info = self._analyze_model(weights, model_path.name)
        
        # Insert model metadata
        cursor.execute('''
            INSERT OR REPLACE INTO models 
            (model_id, model_name, architecture, vocab_size, parameter_count, quality_score, domain, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            model_id,
            model_path.name,
            model_info['architecture'],
            model_info['vocab_size'],
            model_info['parameter_count'],
            model_info['quality_score'],
            model_info['domain'],
            json.dumps(model_info['metadata'])
        ))
        
        # Insert weights
        for key, tensor in weights.items():
            # Serialize tensor
            tensor_bytes = self._serialize_tensor(tensor)
            shape_json = json.dumps(list(tensor.shape))
            
            # Calculate importance score
            importance = self._calculate_importance(key, tensor)
            
            cursor.execute('''
                INSERT INTO weights 
                (model_id, tensor_key, tensor_shape, tensor_dtype, tensor_data, importance_score, fusion_metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                model_id,
                key,
                shape_json,
                str(tensor.dtype),
                tensor_bytes,
                importance,
                json.dumps({'key_pattern': key, 'layer_type': self._classify_layer(key)})
            ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"📥 Imported {model_id}: {len(weights)} tensors")
        return model_id
    
    def query_compatible_weights(self, target_shape: List[int], layer_type: str, 
                               min_quality: float = 0.5) -> List[Dict]:
        """Query database for compatible weights"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        shape_json = json.dumps(target_shape)
        
        cursor.execute('''
            SELECT w.*, m.quality_score, m.domain 
            FROM weights w
            JOIN models m ON w.model_id = m.model_id
            WHERE w.tensor_shape = ? 
            AND json_extract(w.fusion_metadata, '$.layer_type') = ?
            AND m.quality_score >= ?
            ORDER BY (w.importance_score * m.quality_score) DESC
        ''', (shape_json, layer_type, min_quality))
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return results
    
    def create_fusion_recipe(self, model_sources: List[str], strategy: str) -> str:
        """Create a fusion recipe in the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Analyze compatibility and create weight mapping
        weight_mapping = self._analyze_fusion_compatibility(model_sources)
        
        cursor.execute('''
            INSERT INTO fusion_recipes 
            (recipe_name, fusion_strategy, model_sources, weight_mapping)
            VALUES (?, ?, ?, ?)
        ''', (
            f"fusion_{'_'.join(model_sources)}",
            strategy,
            json.dumps(model_sources),
            json.dumps(weight_mapping)
        ))
        
        recipe_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"🍳 Created fusion recipe {recipe_id} for {model_sources}")
        return str(recipe_id)
    
    def execute_fusion(self, recipe_id: str) -> Dict[str, torch.Tensor]:
        """Execute a fusion recipe from the database"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get fusion recipe
        cursor.execute('SELECT * FROM fusion_recipes WHERE recipe_id = ?', (recipe_id,))
        recipe = dict(cursor.fetchone())
        
        model_sources = json.loads(recipe['model_sources'])
        weight_mapping = json.loads(recipe['weight_mapping'])
        strategy = recipe['fusion_strategy']
        
        # Execute fusion based on strategy
        if strategy == 'quality_weighted_average':
            fused_weights = self._quality_weighted_fusion(model_sources, weight_mapping)
        elif strategy == 'domain_specialized':
            fused_weights = self._domain_specialized_fusion(model_sources, weight_mapping)
        elif strategy == 'attention_based':
            fused_weights = self._attention_based_fusion(model_sources, weight_mapping)
        else:
            fused_weights = self._smart_average_fusion(model_sources, weight_mapping)
        
        conn.close()
        return fused_weights
    
    def _quality_weighted_fusion(self, model_sources: List[str], weight_mapping: Dict) -> Dict[str, torch.Tensor]:
        """Database-driven quality-weighted fusion"""
        fused_model = {}
        
        for tensor_key, source_info in weight_mapping.items():
            compatible_weights = []
            quality_weights = []
            
            for model_id in model_sources:
                # Query for this specific weight
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT w.*, m.quality_score 
                    FROM weights w
                    JOIN models m ON w.model_id = m.model_id
                    WHERE w.model_id = ? AND w.tensor_key = ?
                ''', (model_id, tensor_key))
                
                result = cursor.fetchone()
                if result:
                    weight_data = dict(result)
                    tensor = self._deserialize_tensor(
                        weight_data['tensor_data'],
                        json.loads(weight_data['tensor_shape']),
                        weight_data['tensor_dtype']
                    )
                    compatible_weights.append(tensor)
                    quality_weights.append(weight_data['quality_score'])
                
                conn.close()
            
            if compatible_weights:
                # Quality-weighted average
                total_quality = sum(quality_weights)
                normalized_weights = [w / total_quality for w in quality_weights]
                
                fused_tensor = torch.zeros_like(compatible_weights[0])
                for tensor, weight in zip(compatible_weights, normalized_weights):
                    fused_tensor += tensor * weight
                
                fused_model[tensor_key] = fused_tensor
        
        return fused_model
    
    def _serialize_tensor(self, tensor: torch.Tensor) -> bytes:
        """Serialize tensor to bytes for database storage"""
        return tensor.numpy().tobytes()
    
    def _deserialize_tensor(self, data: bytes, shape: List[int], dtype: str) -> torch.Tensor:
        """Deserialize tensor from database bytes"""
        np_array = np.frombuffer(data, dtype=np.float32).reshape(shape)
        return torch.from_numpy(np_array)
    
    def _analyze_model(self, weights: Dict, model_name: str) -> Dict[str, Any]:
        """Analyze model characteristics"""
        # [Implementation similar to previous analyzers]
        return {
            'architecture': 'llama',  # Simplified
            'vocab_size': 32000,
            'parameter_count': sum(t.numel() for t in weights.values()),
            'quality_score': 0.8,
            'domain': 'general',
            'metadata': {'analyzed_at': '2024-01-01'}
        }
    
    def _calculate_importance(self, key: str, tensor: torch.Tensor) -> float:
        """Calculate importance score for a weight tensor"""
        base_score = 0.5
        
        # Embedding layers are important
        if 'embed' in key:
            base_score += 0.3
        
        # Attention layers are important  
        if 'attention' in key:
            base_score += 0.2
        
        # Output layers are important
        if 'output' in key or 'head' in key:
            base_score += 0.2
        
        return min(base_score, 1.0)
    
    def _classify_layer(self, key: str) -> str:
        """Classify layer type from tensor key"""
        if 'embed' in key:
            return 'embedding'
        elif 'attention' in key:
            return 'attention'
        elif 'mlp' in key or 'ffn' in key:
            return 'feedforward'
        elif 'output' in key or 'head' in key:
            return 'output'
        else:
            return 'other'
    
    def _analyze_fusion_compatibility(self, model_sources: List[str]) -> Dict[str, Any]:
        """Analyze compatibility between models for fusion"""
        # Simplified implementation
        return {f"layer_{i}": {"sources": model_sources} for i in range(100)}