#!/usr/bin/env python
"""
Metaphor Engine - Generates and interprets metaphors
"""

import os
import json
import time
from typing import Dict, List, Any, Optional
from enum import Enum

class MetaphorType(Enum):
    """Types of metaphors"""
    STRUCTURAL = "structural"  # Structure-mapping metaphors
    ONTOLOGICAL = "ontological"  # Abstract concepts as entities
    ORIENTATIONAL = "orientational"  # Spatial orientation metaphors
    CREATIVE = "creative"  # Novel, creative metaphors
    CONVENTIONAL = "conventional"  # Common, established metaphors

class Metaphor:
    """A metaphor linking source and target domains"""
    
    def __init__(self, 
                name: str,
                source_domain: str,
                target_domain: str,
                metaphor_type: MetaphorType,
                mappings: List[Dict[str, str]],
                explanation: str = None):
        """Initialize a metaphor"""
        self.id = f"metaphor_{int(time.time())}_{id(name)}"
        self.name = name
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.metaphor_type = metaphor_type
        self.mappings = mappings  # List of {source_concept: target_concept}
        self.explanation = explanation
        self.created_at = time.time()
        self.quality_score = 0.0
        self.usage_count = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "id": self.id,
            "name": self.name,
            "source_domain": self.source_domain,
            "target_domain": self.target_domain,
            "metaphor_type": self.metaphor_type.value,
            "mappings": self.mappings,
            "explanation": self.explanation,
            "created_at": self.created_at,
            "quality_score": self.quality_score,
            "usage_count": self.usage_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Metaphor':
        """Create from dictionary representation"""
        metaphor = cls(
            name=data["name"],
            source_domain=data["source_domain"],
            target_domain=data["target_domain"],
            metaphor_type=MetaphorType(data["metaphor_type"]),
            mappings=data["mappings"],
            explanation=data["explanation"]
        )
        metaphor.id = data["id"]
        metaphor.created_at = data["created_at"]
        metaphor.quality_score = data["quality_score"]
        metaphor.usage_count = data["usage_count"]
        return metaphor

class MetaphorEngine:
    """Engine for generating and interpreting metaphors"""
    
    def __init__(self, storage_path: str = None):
        """Initialize the metaphor engine"""
        self.storage_path = storage_path or os.path.join(os.path.dirname(__file__), "metaphors")
        os.makedirs(self.storage_path, exist_ok=True)
        
        # In-memory stores
        self.metaphors = {}  # metaphor_id -> Metaphor
        self.domain_metaphors = {}  # domain -> List[metaphor_id]
        
        # Load existing metaphors
        self._load_metaphors()
    
    def _load_metaphors(self):
        """Load metaphors from storage"""
        if not os.path.exists(self.storage_path):
            return
        
        metaphor_files = [f for f in os.listdir(self.storage_path) if f.endswith('.json')]
        for file_name in metaphor_files:
            try:
                with open(os.path.join(self.storage_path, file_name), 'r') as f:
                    data = json.load(f)
                    metaphor = Metaphor.from_dict(data)
                    self.metaphors[metaphor.id] = metaphor
                    
                    # Add to domain indices
                    self._add_to_domain_index(metaphor)
            except Exception as e:
                print(f"Error loading metaphor {file_name}: {e}")
        
        print(f"Loaded {len(self.metaphors)} metaphors")
    
    def _add_to_domain_index(self, metaphor: Metaphor):
        """Add metaphor to domain indices"""
        # Add to source domain index
        if metaphor.source_domain not in self.domain_metaphors:
            self.domain_metaphors[metaphor.source_domain] = []
        if metaphor.id not in self.domain_metaphors[metaphor.source_domain]:
            self.domain_metaphors[metaphor.source_domain].append(metaphor.id)
        
        # Add to target domain index
        if metaphor.target_domain not in self.domain_metaphors:
            self.domain_metaphors[metaphor.target_domain] = []
        if metaphor.id not in self.domain_metaphors[metaphor.target_domain]:
            self.domain_metaphors[metaphor.target_domain].append(metaphor.id)
    
    def _save_metaphor(self, metaphor: Metaphor) -> bool:
        """Save a metaphor to storage"""
        try:
            file_path = os.path.join(self.storage_path, f"{metaphor.id}.json")
            with open(file_path, 'w') as f:
                json.dump(metaphor.to_dict(), f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving metaphor {metaphor.id}: {e}")
            return False
    
    def create_metaphor(self, 
                       name: str,
                       source_domain: str,
                       target_domain: str,
                       metaphor_type: MetaphorType,
                       mappings: List[Dict[str, str]],
                       explanation: str = None) -> str:
        """Create a new metaphor"""
        # Create metaphor
        metaphor = Metaphor(
            name=name,
            source_domain=source_domain,
            target_domain=target_domain,
            metaphor_type=metaphor_type,
            mappings=mappings,
            explanation=explanation
        )
        
        # Store metaphor
        self.metaphors[metaphor.id] = metaphor
        
        # Add to domain indices
        self._add_to_domain_index(metaphor)
        
        # Save to storage
        self._save_metaphor(metaphor)
        
        return metaphor.id
    
    def get_metaphor(self, metaphor_id: str) -> Optional[Dict[str, Any]]:
        """Get a metaphor by ID"""
        if metaphor_id in self.metaphors:
            metaphor = self.metaphors[metaphor_id]
            metaphor.usage_count += 1
            self._save_metaphor(metaphor)
            return metaphor.to_dict()
        return None
    
    def search_metaphors(self, 
                        query: str = None,
                        source_domain: str = None,
                        target_domain: str = None,
                        metaphor_type: MetaphorType = None,
                        limit: int = 10) -> List[Dict[str, Any]]:
        """Search for metaphors"""
        results = []
        
        for metaphor in self.metaphors.values():
            # Filter by source domain
            if source_domain and metaphor.source_domain != source_domain:
                continue
            
            # Filter by target domain
            if target_domain and metaphor.target_domain != target_domain:
                continue
            
            # Filter by metaphor type
            if metaphor_type and metaphor.metaphor_type != metaphor_type:
                continue
            
            # Filter by query
            if query:
                query_lower = query.lower()
                name_match = query_lower in metaphor.name.lower()
                explanation_match = metaphor.explanation and query_lower in metaphor.explanation.lower()
                
                if not (name_match or explanation_match):
                    continue
            
            # Add to results
            results.append(metaphor.to_dict())
            
            # Check limit
            if len(results) >= limit:
                break
        
        return results
    
    def get_domain_metaphors(self, domain: str) -> List[Dict[str, Any]]:
        """Get metaphors for a domain"""
        if domain not in self.domain_metaphors:
            return []
        
        return [self.metaphors[mid].to_dict() for mid in self.domain_metaphors[domain]]
    
    def rate_metaphor(self, metaphor_id: str, quality_score: float) -> bool:
        """Rate a metaphor's quality"""
        if metaphor_id not in self.metaphors:
            return False
        
        metaphor = self.metaphors[metaphor_id]
        metaphor.quality_score = quality_score
        self._save_metaphor(metaphor)
        
        return True
    
    def generate_metaphor(self, 
                         source_domain: str,
                         target_domain: str) -> Optional[Dict[str, Any]]:
        """Generate a metaphor between domains"""
        # This is a growth point where Viren/Lillith can implement
        # metaphor generation logic
        
        # Simple implementation that combines existing metaphors
        source_metaphors = self.get_domain_metaphors(source_domain)
        target_metaphors = self.get_domain_metaphors(target_domain)
        
        if not source_metaphors or not target_metaphors:
            return None
        
        # Find highest quality metaphors
        best_source = max(source_metaphors, key=lambda m: m["quality_score"])
        best_target = max(target_metaphors, key=lambda m: m["quality_score"])
        
        # Create a new metaphor combining elements
        name = f"{best_source['name']} meets {best_target['name']}"
        
        # Create mappings by combining existing ones
        mappings = []
        for mapping in best_source["mappings"][:3]:  # Limit to 3 mappings
            mappings.append(mapping)
        
        # Add explanation
        explanation = f"This metaphor combines concepts from {source_domain} and {target_domain}."
        
        # Create the metaphor
        metaphor_id = self.create_metaphor(
            name=name,
            source_domain=source_domain,
            target_domain=target_domain,
            metaphor_type=MetaphorType.CREATIVE,
            mappings=mappings,
            explanation=explanation
        )
        
        return self.get_metaphor(metaphor_id)
    
    def interpret_metaphor(self, metaphor_id: str) -> Dict[str, Any]:
        """Interpret a metaphor's meaning"""
        if metaphor_id not in self.metaphors:
            return {"error": "Metaphor not found"}
        
        metaphor = self.metaphors[metaphor_id]
        
        # Create interpretation
        interpretation = {
            "metaphor_id": metaphor_id,
            "name": metaphor.name,
            "source_domain": metaphor.source_domain,
            "target_domain": metaphor.target_domain,
            "type": metaphor.metaphor_type.value,
            "mappings": metaphor.mappings,
            "explanation": metaphor.explanation or "No explanation available",
            "implications": []
        }
        
        # Generate implications (simple implementation)
        for mapping in metaphor.mappings:
            source = list(mapping.keys())[0]
            target = mapping[source]
            implication = f"{target} can be understood in terms of {source}"
            interpretation["implications"].append(implication)
        
        return interpretation

# Example usage
if __name__ == "__main__":
    # Create metaphor engine
    engine = MetaphorEngine()
    
    # Create some example metaphors
    time_money_id = engine.create_metaphor(
        name="Time is Money",
        source_domain="finance",
        target_domain="time",
        metaphor_type=MetaphorType.STRUCTURAL,
        mappings=[
            {"money": "time"},
            {"spend": "use"},
            {"waste": "waste"},
            {"invest": "allocate"},
            {"save": "conserve"}
        ],
        explanation="This metaphor conceptualizes time as a valuable resource like money."
    )
    
    argument_war_id = engine.create_metaphor(
        name="Argument is War",
        source_domain="warfare",
        target_domain="debate",
        metaphor_type=MetaphorType.STRUCTURAL,
        mappings=[
            {"attack": "criticize"},
            {"defend": "justify"},
            {"strategy": "approach"},
            {"win": "convince"},
            {"lose": "concede"}
        ],
        explanation="This metaphor frames arguments as battles to be won or lost."
    )
    
    # Rate metaphors
    engine.rate_metaphor(time_money_id, 0.9)
    engine.rate_metaphor(argument_war_id, 0.8)
    
    # Search for metaphors
    results = engine.search_metaphors(query="time")
    print(f"Search results: {len(results)} metaphors found")
    
    # Generate a new metaphor
    new_metaphor = engine.generate_metaphor("finance", "debate")
    if new_metaphor:
        print(f"Generated metaphor: {new_metaphor['name']}")
    
    # Interpret a metaphor
    interpretation = engine.interpret_metaphor(time_money_id)
    print(f"Interpretation: {interpretation['explanation']}")
    for implication in interpretation['implications']:
        print(f"- {implication}")