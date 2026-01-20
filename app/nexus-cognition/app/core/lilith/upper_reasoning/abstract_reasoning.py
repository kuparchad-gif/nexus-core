#!/usr/bin/env python
"""
Abstract Reasoning - Framework for analogical reasoning and concept abstraction
"""

import os
import json
import time
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
from pathlib import Path

class AbstractionLevel(Enum):
    """Levels of abstraction in reasoning"""
    CONCRETE = 0
    SPECIFIC = 1
    GENERAL = 2
    ABSTRACT = 3
    UNIVERSAL = 4

class RelationType(Enum):
    """Types of relations between concepts"""
    IS_A = "is_a"  # Taxonomy
    PART_OF = "part_of"  # Meronymy
    SIMILAR_TO = "similar_to"  # Similarity
    OPPOSITE_OF = "opposite_of"  # Antonymy
    CAUSES = "causes"  # Causality
    PRECEDES = "precedes"  # Temporal
    USED_FOR = "used_for"  # Function
    LOCATED_IN = "located_in"  # Spatial
    MADE_OF = "made_of"  # Composition
    DEFINED_AS = "defined_as"  # Definition

class Concept:
    """A concept with properties and relations"""
    
    def __init__(self, 
                name: str, 
                abstraction_level: AbstractionLevel = AbstractionLevel.SPECIFIC,
                properties: Dict[str, Any] = None,
                domain: str = None):
        """Initialize a concept
        
        Args:
            name: Concept name
            abstraction_level: Level of abstraction
            properties: Concept properties
            domain: Knowledge domain
        """
        self.id = f"concept_{int(time.time())}_{id(name)}"
        self.name = name
        self.abstraction_level = abstraction_level
        self.properties = properties or {}
        self.domain = domain
        self.relations = []  # List of (relation_type, target_concept_id)
        self.created_at = time.time()
        self.access_count = 0
        self.confidence = 0.8  # Default confidence
    
    def add_relation(self, relation_type: RelationType, target_concept_id: str, confidence: float = 0.8):
        """Add a relation to another concept
        
        Args:
            relation_type: Type of relation
            target_concept_id: ID of target concept
            confidence: Confidence in the relation (0.0 to 1.0)
        """
        self.relations.append({
            "type": relation_type.value,
            "target": target_concept_id,
            "confidence": confidence
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "id": self.id,
            "name": self.name,
            "abstraction_level": self.abstraction_level.value,
            "properties": self.properties,
            "domain": self.domain,
            "relations": self.relations,
            "created_at": self.created_at,
            "access_count": self.access_count,
            "confidence": self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Concept':
        """Create from dictionary representation"""
        concept = cls(
            name=data["name"],
            abstraction_level=AbstractionLevel(data["abstraction_level"]),
            properties=data["properties"],
            domain=data["domain"]
        )
        concept.id = data["id"]
        concept.relations = data["relations"]
        concept.created_at = data["created_at"]
        concept.access_count = data["access_count"]
        concept.confidence = data["confidence"]
        return concept

class Analogy:
    """An analogy between two concept pairs"""
    
    def __init__(self, 
                source_from_id: str,
                source_to_id: str,
                target_from_id: str,
                target_to_id: Optional[str] = None,
                relation_type: Optional[RelationType] = None,
                confidence: float = 0.5):
        """Initialize an analogy
        
        Args:
            source_from_id: Source domain "from" concept ID
            source_to_id: Source domain "to" concept ID
            target_from_id: Target domain "from" concept ID
            target_to_id: Target domain "to" concept ID (may be None for prediction)
            relation_type: Type of relation in the analogy
            confidence: Confidence in the analogy (0.0 to 1.0)
        """
        self.id = f"analogy_{int(time.time())}_{id(self)}"
        self.source_from_id = source_from_id
        self.source_to_id = source_to_id
        self.target_from_id = target_from_id
        self.target_to_id = target_to_id
        self.relation_type = relation_type.value if relation_type else None
        self.confidence = confidence
        self.created_at = time.time()
        self.validated = False
        self.validation_score = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "id": self.id,
            "source_from_id": self.source_from_id,
            "source_to_id": self.source_to_id,
            "target_from_id": self.target_from_id,
            "target_to_id": self.target_to_id,
            "relation_type": self.relation_type,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "validated": self.validated,
            "validation_score": self.validation_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Analogy':
        """Create from dictionary representation"""
        analogy = cls(
            source_from_id=data["source_from_id"],
            source_to_id=data["source_to_id"],
            target_from_id=data["target_from_id"],
            target_to_id=data["target_to_id"],
            relation_type=RelationType(data["relation_type"]) if data["relation_type"] else None,
            confidence=data["confidence"]
        )
        analogy.id = data["id"]
        analogy.created_at = data["created_at"]
        analogy.validated = data["validated"]
        analogy.validation_score = data["validation_score"]
        return analogy

class AbstractReasoning:
    """Abstract reasoning system for analogical reasoning and concept abstraction"""
    
    def __init__(self, storage_path: str = None):
        """Initialize the abstract reasoning system
        
        Args:
            storage_path: Path to store concepts and analogies
        """
        self.storage_path = storage_path or os.path.join(os.path.dirname(__file__), "abstract_reasoning")
        
        # Create storage directories
        self.concepts_path = os.path.join(self.storage_path, "concepts")
        self.analogies_path = os.path.join(self.storage_path, "analogies")
        self.domains_path = os.path.join(self.storage_path, "domains")
        
        os.makedirs(self.concepts_path, exist_ok=True)
        os.makedirs(self.analogies_path, exist_ok=True)
        os.makedirs(self.domains_path, exist_ok=True)
        
        # In-memory stores
        self.concepts = {}  # concept_id -> Concept
        self.analogies = {}  # analogy_id -> Analogy
        self.domains = {}  # domain_name -> List[concept_id]
        
        # Load existing data
        self._load_data()
    
    def _load_data(self):
        """Load concepts and analogies from storage"""
        # Load concepts
        concept_files = [f for f in os.listdir(self.concepts_path) if f.endswith('.json')]
        for file_name in concept_files:
            try:
                with open(os.path.join(self.concepts_path, file_name), 'r') as f:
                    data = json.load(f)
                    concept = Concept.from_dict(data)
                    self.concepts[concept.id] = concept
                    
                    # Add to domain index
                    if concept.domain:
                        if concept.domain not in self.domains:
                            self.domains[concept.domain] = []
                        self.domains[concept.domain].append(concept.id)
            except Exception as e:
                print(f"Error loading concept {file_name}: {e}")
        
        # Load analogies
        analogy_files = [f for f in os.listdir(self.analogies_path) if f.endswith('.json')]
        for file_name in analogy_files:
            try:
                with open(os.path.join(self.analogies_path, file_name), 'r') as f:
                    data = json.load(f)
                    analogy = Analogy.from_dict(data)
                    self.analogies[analogy.id] = analogy
            except Exception as e:
                print(f"Error loading analogy {file_name}: {e}")
        
        print(f"Loaded {len(self.concepts)} concepts and {len(self.analogies)} analogies")
    
    def _save_concept(self, concept: Concept) -> bool:
        """Save a concept to storage
        
        Args:
            concept: Concept to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = os.path.join(self.concepts_path, f"{concept.id}.json")
            with open(file_path, 'w') as f:
                json.dump(concept.to_dict(), f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving concept {concept.id}: {e}")
            return False
    
    def _save_analogy(self, analogy: Analogy) -> bool:
        """Save an analogy to storage
        
        Args:
            analogy: Analogy to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = os.path.join(self.analogies_path, f"{analogy.id}.json")
            with open(file_path, 'w') as f:
                json.dump(analogy.to_dict(), f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving analogy {analogy.id}: {e}")
            return False
    
    def add_concept(self, 
                   name: str, 
                   properties: Dict[str, Any] = None,
                   domain: str = None,
                   abstraction_level: AbstractionLevel = AbstractionLevel.SPECIFIC) -> str:
        """Add a new concept
        
        Args:
            name: Concept name
            properties: Concept properties
            domain: Knowledge domain
            abstraction_level: Level of abstraction
            
        Returns:
            Concept ID
        """
        # Create concept
        concept = Concept(
            name=name,
            properties=properties,
            domain=domain,
            abstraction_level=abstraction_level
        )
        
        # Store concept
        self.concepts[concept.id] = concept
        
        # Add to domain index
        if domain:
            if domain not in self.domains:
                self.domains[domain] = []
            self.domains[domain].append(concept.id)
        
        # Save to storage
        self._save_concept(concept)
        
        return concept.id
    
    def add_relation(self, 
                    from_concept_id: str, 
                    to_concept_id: str,
                    relation_type: RelationType,
                    confidence: float = 0.8) -> bool:
        """Add a relation between concepts
        
        Args:
            from_concept_id: Source concept ID
            to_concept_id: Target concept ID
            relation_type: Type of relation
            confidence: Confidence in the relation
            
        Returns:
            True if successful, False otherwise
        """
        # Check if concepts exist
        if from_concept_id not in self.concepts or to_concept_id not in self.concepts:
            return False
        
        # Add relation
        from_concept = self.concepts[from_concept_id]
        from_concept.add_relation(relation_type, to_concept_id, confidence)
        
        # Save concept
        self._save_concept(from_concept)
        
        return True
    
    def create_analogy(self, 
                      source_from_id: str,
                      source_to_id: str,
                      target_from_id: str,
                      target_to_id: Optional[str] = None,
                      relation_type: Optional[RelationType] = None,
                      confidence: float = 0.5) -> str:
        """Create an analogy between concept pairs
        
        Args:
            source_from_id: Source domain "from" concept ID
            source_to_id: Source domain "to" concept ID
            target_from_id: Target domain "from" concept ID
            target_to_id: Target domain "to" concept ID (may be None for prediction)
            relation_type: Type of relation in the analogy
            confidence: Confidence in the analogy
            
        Returns:
            Analogy ID
        """
        # Check if concepts exist
        required_concepts = [source_from_id, source_to_id, target_from_id]
        if target_to_id:
            required_concepts.append(target_to_id)
        
        for concept_id in required_concepts:
            if concept_id not in self.concepts:
                return None
        
        # Create analogy
        analogy = Analogy(
            source_from_id=source_from_id,
            source_to_id=source_to_id,
            target_from_id=target_from_id,
            target_to_id=target_to_id,
            relation_type=relation_type,
            confidence=confidence
        )
        
        # Store analogy
        self.analogies[analogy.id] = analogy
        
        # Save to storage
        self._save_analogy(analogy)
        
        return analogy.id
    
    def find_analogies(self, 
                      source_domain: str, 
                      target_domain: str,
                      min_confidence: float = 0.5) -> List[Dict[str, Any]]:
        """Find potential analogies between domains
        
        Args:
            source_domain: Source knowledge domain
            target_domain: Target knowledge domain
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of potential analogies
        """
        results = []
        
        # Check if domains exist
        if source_domain not in self.domains or target_domain not in self.domains:
            return results
        
        # Get concepts from each domain
        source_concepts = [self.concepts[cid] for cid in self.domains[source_domain]]
        target_concepts = [self.concepts[cid] for cid in self.domains[target_domain]]
        
        # Find relations in source domain
        source_relations = []
        for concept in source_concepts:
            for relation in concept.relations:
                if relation["target"] in self.domains[source_domain]:
                    source_relations.append({
                        "from_id": concept.id,
                        "to_id": relation["target"],
                        "type": relation["type"],
                        "confidence": relation["confidence"]
                    })
        
        # Find relations in target domain
        target_relations = []
        for concept in target_concepts:
            for relation in concept.relations:
                if relation["target"] in self.domains[target_domain]:
                    target_relations.append({
                        "from_id": concept.id,
                        "to_id": relation["target"],
                        "type": relation["type"],
                        "confidence": relation["confidence"]
                    })
        
        # Find matching relation types
        for s_rel in source_relations:
            for t_rel in target_relations:
                if s_rel["type"] == t_rel["type"]:
                    # Calculate confidence
                    confidence = s_rel["confidence"] * t_rel["confidence"]
                    
                    if confidence >= min_confidence:
                        # Create analogy
                        analogy_id = self.create_analogy(
                            source_from_id=s_rel["from_id"],
                            source_to_id=s_rel["to_id"],
                            target_from_id=t_rel["from_id"],
                            target_to_id=t_rel["to_id"],
                            relation_type=RelationType(s_rel["type"]),
                            confidence=confidence
                        )
                        
                        if analogy_id:
                            results.append(self.analogies[analogy_id].to_dict())
        
        return results
    
    def predict_by_analogy(self, 
                          source_from_id: str,
                          source_to_id: str,
                          target_from_id: str) -> List[Dict[str, Any]]:
        """Predict a concept by analogy
        
        Args:
            source_from_id: Source domain "from" concept ID
            source_to_id: Source domain "to" concept ID
            target_from_id: Target domain "from" concept ID
            
        Returns:
            List of potential target concepts with confidence scores
        """
        results = []
        
        # Check if concepts exist
        if (source_from_id not in self.concepts or 
            source_to_id not in self.concepts or 
            target_from_id not in self.concepts):
            return results
        
        # Get source concepts
        source_from = self.concepts[source_from_id]
        source_to = self.concepts[source_to_id]
        target_from = self.concepts[target_from_id]
        
        # Find relation type between source concepts
        relation_type = None
        relation_confidence = 0.0
        
        for relation in source_from.relations:
            if relation["target"] == source_to_id:
                relation_type = relation["type"]
                relation_confidence = relation["confidence"]
                break
        
        if not relation_type:
            return results
        
        # Find target concepts with same relation
        for relation in target_from.relations:
            if relation["type"] == relation_type and relation["target"] in self.concepts:
                target_to = self.concepts[relation["target"]]
                
                # Calculate confidence
                confidence = relation_confidence * relation["confidence"]
                
                # Add to results
                results.append({
                    "concept_id": target_to.id,
                    "name": target_to.name,
                    "confidence": confidence,
                    "relation_type": relation_type
                })
        
        # Sort by confidence
        results.sort(key=lambda x: x["confidence"], reverse=True)
        
        return results
    
    def abstract_concept(self, concept_id: str) -> Optional[str]:
        """Create a more abstract version of a concept
        
        Args:
            concept_id: Concept ID
            
        Returns:
            ID of the new abstract concept, or None if failed
        """
        # Check if concept exists
        if concept_id not in self.concepts:
            return None
        
        concept = self.concepts[concept_id]
        
        # Skip if already at highest abstraction level
        if concept.abstraction_level == AbstractionLevel.UNIVERSAL:
            return None
        
        # Create new abstraction level
        new_level = AbstractionLevel(min(concept.abstraction_level.value + 1, AbstractionLevel.UNIVERSAL.value))
        
        # Create abstract properties
        abstract_properties = {}
        for key, value in concept.properties.items():
            # Keep only essential properties
            if isinstance(value, (int, float)):
                # For numerical properties, use ranges
                abstract_properties[key] = {"min": value * 0.8, "max": value * 1.2}
            elif isinstance(value, str):
                # For string properties, keep as is if short
                if len(value) < 20:
                    abstract_properties[key] = value
            elif isinstance(value, bool):
                # Keep boolean properties
                abstract_properties[key] = value
        
        # Create abstract concept
        abstract_name = f"Abstract {concept.name}"
        abstract_id = self.add_concept(
            name=abstract_name,
            properties=abstract_properties,
            domain=concept.domain,
            abstraction_level=new_level
        )
        
        # Add IS_A relation from original to abstract
        self.add_relation(
            from_concept_id=concept_id,
            to_concept_id=abstract_id,
            relation_type=RelationType.IS_A,
            confidence=0.9
        )
        
        return abstract_id
    
    def extend_analogy_domains(self, source_domain: str, target_domain: str) -> Dict[str, Any]:
        """Extend analogy domains by finding new connections
        
        Args:
            source_domain: Source knowledge domain
            target_domain: Target knowledge domain
            
        Returns:
            Dictionary with extension results
        """
        # This is a growth point where Viren/Lillith can implement
        # domain extension logic
        
        # Simple implementation to find new analogies
        analogies = self.find_analogies(source_domain, target_domain, min_confidence=0.3)
        
        # Count new analogies by type
        analogy_types = {}
        for analogy in analogies:
            rel_type = analogy.get("relation_type")
            if rel_type:
                if rel_type not in analogy_types:
                    analogy_types[rel_type] = 0
                analogy_types[rel_type] += 1
        
        return {
            "source_domain": source_domain,
            "target_domain": target_domain,
            "new_analogies": len(analogies),
            "analogy_types": analogy_types
        }
    
    def get_concept(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """Get a concept by ID
        
        Args:
            concept_id: Concept ID
            
        Returns:
            Concept dictionary or None if not found
        """
        if concept_id in self.concepts:
            concept = self.concepts[concept_id]
            concept.access_count += 1
            return concept.to_dict()
        return None
    
    def get_analogy(self, analogy_id: str) -> Optional[Dict[str, Any]]:
        """Get an analogy by ID
        
        Args:
            analogy_id: Analogy ID
            
        Returns:
            Analogy dictionary or None if not found
        """
        if analogy_id in self.analogies:
            return self.analogies[analogy_id].to_dict()
        return None
    
    def search_concepts(self, 
                       query: str = None,
                       domain: str = None,
                       abstraction_level: AbstractionLevel = None,
                       limit: int = 10) -> List[Dict[str, Any]]:
        """Search for concepts
        
        Args:
            query: Search query
            domain: Filter by domain
            abstraction_level: Filter by abstraction level
            limit: Maximum number of results
            
        Returns:
            List of matching concepts
        """
        results = []
        
        for concept in self.concepts.values():
            # Filter by domain
            if domain and concept.domain != domain:
                continue
            
            # Filter by abstraction level
            if abstraction_level and concept.abstraction_level != abstraction_level:
                continue
            
            # Filter by query
            if query:
                query_lower = query.lower()
                name_match = query_lower in concept.name.lower()
                property_match = any(
                    query_lower in str(v).lower() 
                    for v in concept.properties.values()
                )
                
                if not (name_match or property_match):
                    continue
            
            # Add to results
            results.append(concept.to_dict())
            
            # Check limit
            if len(results) >= limit:
                break
        
        return results

# Example usage
if __name__ == "__main__":
    # Create abstract reasoning system
    reasoning = AbstractReasoning()
    
    # Add some example concepts
    water_id = reasoning.add_concept(
        name="Water",
        properties={"state": "liquid", "temperature": 20, "essential": True},
        domain="natural_elements"
    )
    
    ice_id = reasoning.add_concept(
        name="Ice",
        properties={"state": "solid", "temperature": -10, "essential": True},
        domain="natural_elements"
    )
    
    steam_id = reasoning.add_concept(
        name="Steam",
        properties={"state": "gas", "temperature": 100, "essential": True},
        domain="natural_elements"
    )
    
    knowledge_id = reasoning.add_concept(
        name="Knowledge",
        properties={"tangible": False, "valuable": True},
        domain="abstract_concepts"
    )
    
    information_id = reasoning.add_concept(
        name="Information",
        properties={"tangible": False, "structured": True},
        domain="abstract_concepts"
    )
    
    wisdom_id = reasoning.add_concept(
        name="Wisdom",
        properties={"tangible": False, "valuable": True, "applied": True},
        domain="abstract_concepts"
    )
    
    # Add relations
    reasoning.add_relation(water_id, ice_id, RelationType.CAUSES, 0.9)
    reasoning.add_relation(water_id, steam_id, RelationType.CAUSES, 0.9)
    reasoning.add_relation(information_id, knowledge_id, RelationType.CAUSES, 0.8)
    reasoning.add_relation(knowledge_id, wisdom_id, RelationType.CAUSES, 0.7)
    
    # Find analogies
    analogies = reasoning.find_analogies("natural_elements", "abstract_concepts")
    print(f"Found {len(analogies)} analogies")
    
    # Predict by analogy
    predictions = reasoning.predict_by_analogy(water_id, ice_id, information_id)
    print(f"Predictions: {predictions}")
    
    # Create abstract concept
    abstract_id = reasoning.abstract_concept(water_id)
    print(f"Created abstract concept: {abstract_id}")
    
    # Extend analogy domains
    extension = reasoning.extend_analogy_domains("natural_elements", "abstract_concepts")
    print(f"Domain extension: {extension}")