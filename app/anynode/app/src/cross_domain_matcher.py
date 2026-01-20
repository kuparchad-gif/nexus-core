#!/usr/bin/env python
"""
Cross-Domain Pattern Matcher - Identifies similar patterns across different domains
"""

import os
import json
import time
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
from pathlib import Path

class PatternType(Enum):
    """Types of patterns that can be matched across domains"""
    STRUCTURAL = "structural"  # Similar structure/organization
    FUNCTIONAL = "functional"  # Similar function/purpose
    CAUSAL = "causal"  # Similar cause-effect relationships
    TEMPORAL = "temporal"  # Similar time-based patterns
    HIERARCHICAL = "hierarchical"  # Similar hierarchical organization
    CYCLICAL = "cyclical"  # Similar repeating cycles
    TRANSFORMATIONAL = "transformational"  # Similar transformation processes

class SimilarityMetric(Enum):
    """Metrics for measuring similarity between patterns"""
    COSINE = "cosine"  # Cosine similarity
    EUCLIDEAN = "euclidean"  # Euclidean distance
    JACCARD = "jaccard"  # Jaccard similarity
    EDIT_DISTANCE = "edit_distance"  # Edit distance
    SEMANTIC = "semantic"  # Semantic similarity

class Pattern:
    """A pattern that can be matched across domains"""
    
    def __init__(self, 
                name: str,
                pattern_type: PatternType,
                domain: str,
                elements: List[Dict[str, Any]],
                metadata: Dict[str, Any] = None):
        """Initialize a pattern
        
        Args:
            name: Pattern name
            pattern_type: Type of pattern
            domain: Knowledge domain
            elements: Pattern elements
            metadata: Additional metadata
        """
        self.id = f"pattern_{int(time.time())}_{id(name)}"
        self.name = name
        self.pattern_type = pattern_type
        self.domain = domain
        self.elements = elements
        self.metadata = metadata or {}
        self.created_at = time.time()
        self.vector = None  # For vector representation
        self.matches = []  # List of matched pattern IDs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "id": self.id,
            "name": self.name,
            "pattern_type": self.pattern_type.value,
            "domain": self.domain,
            "elements": self.elements,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "matches": self.matches,
            "vector": self.vector.tolist() if isinstance(self.vector, np.ndarray) else self.vector
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Pattern':
        """Create from dictionary representation"""
        pattern = cls(
            name=data["name"],
            pattern_type=PatternType(data["pattern_type"]),
            domain=data["domain"],
            elements=data["elements"],
            metadata=data["metadata"]
        )
        pattern.id = data["id"]
        pattern.created_at = data["created_at"]
        pattern.matches = data["matches"]
        if data["vector"] is not None:
            pattern.vector = np.array(data["vector"])
        return pattern

class PatternMatch:
    """A match between two patterns from different domains"""
    
    def __init__(self, 
                pattern1_id: str,
                pattern2_id: str,
                similarity_score: float,
                similarity_metric: SimilarityMetric,
                matching_elements: List[Tuple[int, int]] = None):
        """Initialize a pattern match
        
        Args:
            pattern1_id: First pattern ID
            pattern2_id: Second pattern ID
            similarity_score: Similarity score (0.0 to 1.0)
            similarity_metric: Metric used for similarity
            matching_elements: List of matching element indices
        """
        self.id = f"match_{int(time.time())}_{id(self)}"
        self.pattern1_id = pattern1_id
        self.pattern2_id = pattern2_id
        self.similarity_score = similarity_score
        self.similarity_metric = similarity_metric
        self.matching_elements = matching_elements or []
        self.created_at = time.time()
        self.validated = False
        self.validation_score = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "id": self.id,
            "pattern1_id": self.pattern1_id,
            "pattern2_id": self.pattern2_id,
            "similarity_score": self.similarity_score,
            "similarity_metric": self.similarity_metric.value,
            "matching_elements": self.matching_elements,
            "created_at": self.created_at,
            "validated": self.validated,
            "validation_score": self.validation_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PatternMatch':
        """Create from dictionary representation"""
        match = cls(
            pattern1_id=data["pattern1_id"],
            pattern2_id=data["pattern2_id"],
            similarity_score=data["similarity_score"],
            similarity_metric=SimilarityMetric(data["similarity_metric"]),
            matching_elements=data["matching_elements"]
        )
        match.id = data["id"]
        match.created_at = data["created_at"]
        match.validated = data["validated"]
        match.validation_score = data["validation_score"]
        return match

class CrossDomainMatcher:
    """System for matching patterns across different domains"""
    
    def __init__(self, storage_path: str = None):
        """Initialize the cross-domain matcher
        
        Args:
            storage_path: Path to store patterns and matches
        """
        self.storage_path = storage_path or os.path.join(os.path.dirname(__file__), "cross_domain")
        
        # Create storage directories
        self.patterns_path = os.path.join(self.storage_path, "patterns")
        self.matches_path = os.path.join(self.storage_path, "matches")
        self.domains_path = os.path.join(self.storage_path, "domains")
        
        os.makedirs(self.patterns_path, exist_ok=True)
        os.makedirs(self.matches_path, exist_ok=True)
        os.makedirs(self.domains_path, exist_ok=True)
        
        # In-memory stores
        self.patterns = {}  # pattern_id -> Pattern
        self.matches = {}  # match_id -> PatternMatch
        self.domains = {}  # domain_name -> List[pattern_id]
        
        # Load existing data
        self._load_data()
    
    def _load_data(self):
        """Load patterns and matches from storage"""
        # Load patterns
        pattern_files = [f for f in os.listdir(self.patterns_path) if f.endswith('.json')]
        for file_name in pattern_files:
            try:
                with open(os.path.join(self.patterns_path, file_name), 'r') as f:
                    data = json.load(f)
                    pattern = Pattern.from_dict(data)
                    self.patterns[pattern.id] = pattern
                    
                    # Add to domain index
                    if pattern.domain:
                        if pattern.domain not in self.domains:
                            self.domains[pattern.domain] = []
                        self.domains[pattern.domain].append(pattern.id)
            except Exception as e:
                print(f"Error loading pattern {file_name}: {e}")
        
        # Load matches
        match_files = [f for f in os.listdir(self.matches_path) if f.endswith('.json')]
        for file_name in match_files:
            try:
                with open(os.path.join(self.matches_path, file_name), 'r') as f:
                    data = json.load(f)
                    match = PatternMatch.from_dict(data)
                    self.matches[match.id] = match
            except Exception as e:
                print(f"Error loading match {file_name}: {e}")
        
        print(f"Loaded {len(self.patterns)} patterns and {len(self.matches)} matches")
    
    def _save_pattern(self, pattern: Pattern) -> bool:
        """Save a pattern to storage
        
        Args:
            pattern: Pattern to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = os.path.join(self.patterns_path, f"{pattern.id}.json")
            with open(file_path, 'w') as f:
                json.dump(pattern.to_dict(), f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving pattern {pattern.id}: {e}")
            return False
    
    def _save_match(self, match: PatternMatch) -> bool:
        """Save a match to storage
        
        Args:
            match: Match to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = os.path.join(self.matches_path, f"{match.id}.json")
            with open(file_path, 'w') as f:
                json.dump(match.to_dict(), f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving match {match.id}: {e}")
            return False
    
    def add_pattern(self, 
                   name: str,
                   pattern_type: PatternType,
                   domain: str,
                   elements: List[Dict[str, Any]],
                   metadata: Dict[str, Any] = None) -> str:
        """Add a new pattern
        
        Args:
            name: Pattern name
            pattern_type: Type of pattern
            domain: Knowledge domain
            elements: Pattern elements
            metadata: Additional metadata
            
        Returns:
            Pattern ID
        """
        # Create pattern
        pattern = Pattern(
            name=name,
            pattern_type=pattern_type,
            domain=domain,
            elements=elements,
            metadata=metadata
        )
        
        # Generate vector representation
        pattern.vector = self._vectorize_pattern(pattern)
        
        # Store pattern
        self.patterns[pattern.id] = pattern
        
        # Add to domain index
        if domain:
            if domain not in self.domains:
                self.domains[domain] = []
            self.domains[domain].append(pattern.id)
        
        # Save to storage
        self._save_pattern(pattern)
        
        return pattern.id
    
    def _vectorize_pattern(self, pattern: Pattern) -> np.ndarray:
        """Generate vector representation of a pattern
        
        Args:
            pattern: Pattern to vectorize
            
        Returns:
            Vector representation
        """
        # Simple vectorization based on pattern type and element count
        # In a real implementation, this would use more sophisticated techniques
        
        # Create a basic feature vector
        features = []
        
        # Pattern type (one-hot encoding)
        pattern_types = [pt.value for pt in PatternType]
        type_vector = [1 if pt == pattern.pattern_type.value else 0 for pt in pattern_types]
        features.extend(type_vector)
        
        # Element count
        features.append(len(pattern.elements))
        
        # Element type counts
        element_types = {}
        for element in pattern.elements:
            element_type = element.get("type", "unknown")
            if element_type not in element_types:
                element_types[element_type] = 0
            element_types[element_type] += 1
        
        # Add common element types
        common_types = ["node", "edge", "sequence", "cycle", "branch"]
        for t in common_types:
            features.append(element_types.get(t, 0))
        
        # Convert to numpy array
        return np.array(features, dtype=float)
    
    def find_matches(self, 
                    pattern_id: str, 
                    target_domain: Optional[str] = None,
                    min_similarity: float = 0.7,
                    metric: SimilarityMetric = SimilarityMetric.COSINE) -> List[Dict[str, Any]]:
        """Find matches for a pattern
        
        Args:
            pattern_id: Pattern ID
            target_domain: Target domain (if None, search all domains)
            min_similarity: Minimum similarity threshold
            metric: Similarity metric to use
            
        Returns:
            List of matches
        """
        results = []
        
        # Check if pattern exists
        if pattern_id not in self.patterns:
            return results
        
        source_pattern = self.patterns[pattern_id]
        source_domain = source_pattern.domain
        
        # Get target patterns
        target_patterns = []
        if target_domain:
            # Get patterns from specific domain
            if target_domain in self.domains:
                target_patterns = [self.patterns[pid] for pid in self.domains[target_domain]]
        else:
            # Get patterns from all domains except source domain
            for domain, pattern_ids in self.domains.items():
                if domain != source_domain:
                    target_patterns.extend([self.patterns[pid] for pid in pattern_ids])
        
        # Calculate similarities
        for target_pattern in target_patterns:
            # Skip if same pattern
            if target_pattern.id == pattern_id:
                continue
            
            # Calculate similarity
            similarity, matching_elements = self._calculate_similarity(
                source_pattern, target_pattern, metric
            )
            
            # Check threshold
            if similarity >= min_similarity:
                # Create match
                match = PatternMatch(
                    pattern1_id=pattern_id,
                    pattern2_id=target_pattern.id,
                    similarity_score=similarity,
                    similarity_metric=metric,
                    matching_elements=matching_elements
                )
                
                # Store match
                self.matches[match.id] = match
                
                # Update pattern matches
                source_pattern.matches.append(target_pattern.id)
                target_pattern.matches.append(pattern_id)
                
                # Save match and updated patterns
                self._save_match(match)
                self._save_pattern(source_pattern)
                self._save_pattern(target_pattern)
                
                # Add to results
                results.append(match.to_dict())
        
        return results
    
    def _calculate_similarity(self, 
                             pattern1: Pattern, 
                             pattern2: Pattern,
                             metric: SimilarityMetric) -> Tuple[float, List[Tuple[int, int]]]:
        """Calculate similarity between two patterns
        
        Args:
            pattern1: First pattern
            pattern2: Second pattern
            metric: Similarity metric to use
            
        Returns:
            Tuple of (similarity score, matching elements)
        """
        # Check if vectors are available
        if pattern1.vector is None or pattern2.vector is None:
            return 0.0, []
        
        # Calculate similarity based on metric
        if metric == SimilarityMetric.COSINE:
            # Cosine similarity
            dot_product = np.dot(pattern1.vector, pattern2.vector)
            norm1 = np.linalg.norm(pattern1.vector)
            norm2 = np.linalg.norm(pattern2.vector)
            
            if norm1 == 0 or norm2 == 0:
                similarity = 0.0
            else:
                similarity = dot_product / (norm1 * norm2)
        
        elif metric == SimilarityMetric.EUCLIDEAN:
            # Euclidean distance (converted to similarity)
            distance = np.linalg.norm(pattern1.vector - pattern2.vector)
            max_distance = np.sqrt(len(pattern1.vector))  # Maximum possible distance
            similarity = 1.0 - (distance / max_distance)
        
        elif metric == SimilarityMetric.JACCARD:
            # Jaccard similarity (for binary vectors)
            binary1 = pattern1.vector > 0
            binary2 = pattern2.vector > 0
            
            intersection = np.logical_and(binary1, binary2).sum()
            union = np.logical_or(binary1, binary2).sum()
            
            if union == 0:
                similarity = 0.0
            else:
                similarity = intersection / union
        
        else:
            # Default to simple dot product
            similarity = np.dot(pattern1.vector, pattern2.vector) / (len(pattern1.vector) * 10)
        
        # Find matching elements
        matching_elements = []
        
        # Simple element matching based on type and properties
        for i, elem1 in enumerate(pattern1.elements):
            for j, elem2 in enumerate(pattern2.elements):
                # Check if elements have same type
                if elem1.get("type") == elem2.get("type"):
                    # Check if elements have similar properties
                    prop_match = False
                    
                    # Check common properties
                    for key in set(elem1.keys()) & set(elem2.keys()):
                        if key != "type" and elem1[key] == elem2[key]:
                            prop_match = True
                            break
                    
                    if prop_match:
                        matching_elements.append((i, j))
        
        return similarity, matching_elements
    
    def validate_match(self, 
                      match_id: str, 
                      validation_score: float) -> bool:
        """Validate a pattern match
        
        Args:
            match_id: Match ID
            validation_score: Validation score (0.0 to 1.0)
            
        Returns:
            True if successful, False otherwise
        """
        # Check if match exists
        if match_id not in self.matches:
            return False
        
        # Update match
        match = self.matches[match_id]
        match.validated = True
        match.validation_score = validation_score
        
        # Save match
        self._save_match(match)
        
        return True
    
    def get_pattern(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """Get a pattern by ID
        
        Args:
            pattern_id: Pattern ID
            
        Returns:
            Pattern dictionary or None if not found
        """
        if pattern_id in self.patterns:
            return self.patterns[pattern_id].to_dict()
        return None
    
    def get_match(self, match_id: str) -> Optional[Dict[str, Any]]:
        """Get a match by ID
        
        Args:
            match_id: Match ID
            
        Returns:
            Match dictionary or None if not found
        """
        if match_id in self.matches:
            return self.matches[match_id].to_dict()
        return None
    
    def search_patterns(self, 
                       query: str = None,
                       pattern_type: PatternType = None,
                       domain: str = None,
                       limit: int = 10) -> List[Dict[str, Any]]:
        """Search for patterns
        
        Args:
            query: Search query
            pattern_type: Filter by pattern type
            domain: Filter by domain
            limit: Maximum number of results
            
        Returns:
            List of matching patterns
        """
        results = []
        
        for pattern in self.patterns.values():
            # Filter by pattern type
            if pattern_type and pattern.pattern_type != pattern_type:
                continue
            
            # Filter by domain
            if domain and pattern.domain != domain:
                continue
            
            # Filter by query
            if query:
                query_lower = query.lower()
                name_match = query_lower in pattern.name.lower()
                
                # Check metadata
                metadata_match = False
                for k, v in pattern.metadata.items():
                    if query_lower in str(v).lower():
                        metadata_match = True
                        break
                
                if not (name_match or metadata_match):
                    continue
            
            # Add to results
            results.append(pattern.to_dict())
            
            # Check limit
            if len(results) >= limit:
                break
        
        return results
    
    def get_domain_patterns(self, domain: str) -> List[Dict[str, Any]]:
        """Get all patterns in a domain
        
        Args:
            domain: Domain name
            
        Returns:
            List of patterns
        """
        if domain not in self.domains:
            return []
        
        return [self.patterns[pid].to_dict() for pid in self.domains[domain]]
    
    def get_domains(self) -> List[str]:
        """Get all domains
        
        Returns:
            List of domain names
        """
        return list(self.domains.keys())
    
    def create_similarity_matrix(self, domains: List[str]) -> Dict[str, Any]:
        """Create similarity matrix between domains
        
        Args:
            domains: List of domains
            
        Returns:
            Dictionary with similarity matrix
        """
        # Check if domains exist
        valid_domains = [d for d in domains if d in self.domains]
        
        # Create matrix
        matrix = np.zeros((len(valid_domains), len(valid_domains)))
        
        # Calculate similarities
        for i, domain1 in enumerate(valid_domains):
            for j, domain2 in enumerate(valid_domains):
                if i == j:
                    matrix[i, j] = 1.0  # Self-similarity
                else:
                    # Get patterns from each domain
                    patterns1 = [self.patterns[pid] for pid in self.domains[domain1]]
                    patterns2 = [self.patterns[pid] for pid in self.domains[domain2]]
                    
                    # Calculate average similarity
                    similarities = []
                    for p1 in patterns1:
                        for p2 in patterns2:
                            if p1.vector is not None and p2.vector is not None:
                                sim, _ = self._calculate_similarity(
                                    p1, p2, SimilarityMetric.COSINE
                                )
                                similarities.append(sim)
                    
                    if similarities:
                        matrix[i, j] = sum(similarities) / len(similarities)
        
        return {
            "domains": valid_domains,
            "matrix": matrix.tolist()
        }

# Example usage
if __name__ == "__main__":
    # Create cross-domain matcher
    matcher = CrossDomainMatcher()
    
    # Add some example patterns
    tree_pattern_id = matcher.add_pattern(
        name="Tree Structure",
        pattern_type=PatternType.HIERARCHICAL,
        domain="computer_science",
        elements=[
            {"type": "node", "role": "root", "children": [1, 2]},
            {"type": "node", "role": "branch", "children": [3, 4]},
            {"type": "node", "role": "branch", "children": [5]},
            {"type": "node", "role": "leaf"},
            {"type": "node", "role": "leaf"},
            {"type": "node", "role": "leaf"}
        ],
        metadata={"description": "Tree data structure with root, branches, and leaves"}
    )
    
    family_pattern_id = matcher.add_pattern(
        name="Family Tree",
        pattern_type=PatternType.HIERARCHICAL,
        domain="genealogy",
        elements=[
            {"type": "node", "role": "grandparent", "children": [1, 2]},
            {"type": "node", "role": "parent", "children": [3, 4]},
            {"type": "node", "role": "parent", "children": [5]},
            {"type": "node", "role": "child"},
            {"type": "node", "role": "child"},
            {"type": "node", "role": "child"}
        ],
        metadata={"description": "Family tree with grandparents, parents, and children"}
    )
    
    water_cycle_id = matcher.add_pattern(
        name="Water Cycle",
        pattern_type=PatternType.CYCLICAL,
        domain="natural_science",
        elements=[
            {"type": "state", "name": "evaporation", "next": 1},
            {"type": "state", "name": "condensation", "next": 2},
            {"type": "state", "name": "precipitation", "next": 3},
            {"type": "state", "name": "collection", "next": 0}
        ],
        metadata={"description": "Cycle of water through evaporation, condensation, precipitation, and collection"}
    )
    
    project_cycle_id = matcher.add_pattern(
        name="Project Lifecycle",
        pattern_type=PatternType.CYCLICAL,
        domain="project_management",
        elements=[
            {"type": "state", "name": "planning", "next": 1},
            {"type": "state", "name": "execution", "next": 2},
            {"type": "state", "name": "monitoring", "next": 3},
            {"type": "state", "name": "closure", "next": 0}
        ],
        metadata={"description": "Cycle of project phases through planning, execution, monitoring, and closure"}
    )
    
    # Find matches
    tree_matches = matcher.find_matches(tree_pattern_id)
    print(f"Found {len(tree_matches)} matches for tree pattern")
    
    cycle_matches = matcher.find_matches(water_cycle_id)
    print(f"Found {len(cycle_matches)} matches for water cycle pattern")
    
    # Create similarity matrix
    matrix = matcher.create_similarity_matrix(["computer_science", "genealogy", "natural_science", "project_management"])
    print(f"Similarity matrix: {matrix}")