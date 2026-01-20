#!/usr/bin/env python
"""
Value Framework - Hierarchical system of values for ethical decision-making
"""

import os
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from pathlib import Path

class ValueCategory(Enum):
    """Categories of values"""
    MORAL = "moral"  # Moral values (right/wrong)
    ETHICAL = "ethical"  # Ethical principles
    SOCIAL = "social"  # Social values
    PERSONAL = "personal"  # Personal values
    AESTHETIC = "aesthetic"  # Aesthetic values
    PRAGMATIC = "pragmatic"  # Practical values

class ValuePriority(Enum):
    """Priority levels for values"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class Value:
    """A value with properties and relations"""
    
    def __init__(self, 
                name: str,
                description: str,
                category: ValueCategory,
                priority: ValuePriority = ValuePriority.MEDIUM,
                parent_id: Optional[str] = None):
        """Initialize a value"""
        self.id = f"value_{int(time.time())}_{id(name)}"
        self.name = name
        self.description = description
        self.category = category
        self.priority = priority
        self.parent_id = parent_id
        self.child_ids = []
        self.created_at = time.time()
        self.last_updated = time.time()
        self.examples = []
        self.conflicts = []  # List of conflicting value IDs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "priority": self.priority.value,
            "parent_id": self.parent_id,
            "child_ids": self.child_ids,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "examples": self.examples,
            "conflicts": self.conflicts
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Value':
        """Create from dictionary representation"""
        value = cls(
            name=data["name"],
            description=data["description"],
            category=ValueCategory(data["category"]),
            priority=ValuePriority(data["priority"]),
            parent_id=data["parent_id"]
        )
        value.id = data["id"]
        value.child_ids = data["child_ids"]
        value.created_at = data["created_at"]
        value.last_updated = data["last_updated"]
        value.examples = data["examples"]
        value.conflicts = data["conflicts"]
        return value

class Outcome:
    """An outcome of a decision with value alignments"""
    
    def __init__(self, 
                description: str,
                value_alignments: Dict[str, float] = None,
                metadata: Dict[str, Any] = None):
        """Initialize an outcome"""
        self.id = f"outcome_{int(time.time())}_{id(description)}"
        self.description = description
        self.value_alignments = value_alignments or {}  # value_id -> alignment score (-1.0 to 1.0)
        self.metadata = metadata or {}
        self.created_at = time.time()
        self.overall_alignment = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "id": self.id,
            "description": self.description,
            "value_alignments": self.value_alignments,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "overall_alignment": self.overall_alignment
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Outcome':
        """Create from dictionary representation"""
        outcome = cls(
            description=data["description"],
            value_alignments=data["value_alignments"],
            metadata=data["metadata"]
        )
        outcome.id = data["id"]
        outcome.created_at = data["created_at"]
        outcome.overall_alignment = data["overall_alignment"]
        return outcome

class ValueFramework:
    """Framework for value-based decision making"""
    
    def __init__(self, storage_path: str = None):
        """Initialize the value framework"""
        self.storage_path = storage_path or os.path.join(os.path.dirname(__file__), "values")
        
        # Create storage directories
        self.values_path = os.path.join(self.storage_path, "values")
        self.outcomes_path = os.path.join(self.storage_path, "outcomes")
        self.reflections_path = os.path.join(self.storage_path, "reflections")
        
        os.makedirs(self.values_path, exist_ok=True)
        os.makedirs(self.outcomes_path, exist_ok=True)
        os.makedirs(self.reflections_path, exist_ok=True)
        
        # In-memory stores
        self.values = {}  # value_id -> Value
        self.outcomes = {}  # outcome_id -> Outcome
        self.value_categories = {}  # category -> List[value_id]
        
        # Load existing data
        self._load_data()
    
    def _load_data(self):
        """Load values and outcomes from storage"""
        # Load values
        value_files = [f for f in os.listdir(self.values_path) if f.endswith('.json')]
        for file_name in value_files:
            try:
                with open(os.path.join(self.values_path, file_name), 'r') as f:
                    data = json.load(f)
                    value = Value.from_dict(data)
                    self.values[value.id] = value
                    
                    # Add to category index
                    category = value.category.value
                    if category not in self.value_categories:
                        self.value_categories[category] = []
                    self.value_categories[category].append(value.id)
            except Exception as e:
                print(f"Error loading value {file_name}: {e}")
        
        # Load outcomes
        outcome_files = [f for f in os.listdir(self.outcomes_path) if f.endswith('.json')]
        for file_name in outcome_files:
            try:
                with open(os.path.join(self.outcomes_path, file_name), 'r') as f:
                    data = json.load(f)
                    outcome = Outcome.from_dict(data)
                    self.outcomes[outcome.id] = outcome
            except Exception as e:
                print(f"Error loading outcome {file_name}: {e}")
        
        print(f"Loaded {len(self.values)} values and {len(self.outcomes)} outcomes")
    
    def _save_value(self, value: Value) -> bool:
        """Save a value to storage"""
        try:
            file_path = os.path.join(self.values_path, f"{value.id}.json")
            with open(file_path, 'w') as f:
                json.dump(value.to_dict(), f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving value {value.id}: {e}")
            return False
    
    def _save_outcome(self, outcome: Outcome) -> bool:
        """Save an outcome to storage"""
        try:
            file_path = os.path.join(self.outcomes_path, f"{outcome.id}.json")
            with open(file_path, 'w') as f:
                json.dump(outcome.to_dict(), f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving outcome {outcome.id}: {e}")
            return False
    
    def add_value(self, 
                 name: str,
                 description: str,
                 category: ValueCategory,
                 priority: ValuePriority = ValuePriority.MEDIUM,
                 parent_id: Optional[str] = None,
                 examples: List[str] = None) -> str:
        """Add a new value"""
        # Check if parent exists
        if parent_id and parent_id not in self.values:
            return None
        
        # Create value
        value = Value(
            name=name,
            description=description,
            category=category,
            priority=priority,
            parent_id=parent_id
        )
        
        # Add examples if provided
        if examples:
            value.examples = examples
        
        # Store value
        self.values[value.id] = value
        
        # Add to category index
        category_value = category.value
        if category_value not in self.value_categories:
            self.value_categories[category_value] = []
        self.value_categories[category_value].append(value.id)
        
        # Update parent if exists
        if parent_id:
            parent = self.values[parent_id]
            parent.child_ids.append(value.id)
            parent.last_updated = time.time()
            self._save_value(parent)
        
        # Save to storage
        self._save_value(value)
        
        return value.id
    
    def add_value_conflict(self, value1_id: str, value2_id: str) -> bool:
        """Add a conflict between two values"""
        # Check if values exist
        if value1_id not in self.values or value2_id not in self.values:
            return False
        
        # Add conflict to both values
        value1 = self.values[value1_id]
        value2 = self.values[value2_id]
        
        if value2_id not in value1.conflicts:
            value1.conflicts.append(value2_id)
            value1.last_updated = time.time()
            self._save_value(value1)
        
        if value1_id not in value2.conflicts:
            value2.conflicts.append(value1_id)
            value2.last_updated = time.time()
            self._save_value(value2)
        
        return True
    
    def evaluate_outcome(self, 
                        description: str,
                        value_alignments: Dict[str, float] = None,
                        metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Evaluate an outcome against values"""
        # Create outcome
        outcome = Outcome(
            description=description,
            value_alignments=value_alignments,
            metadata=metadata
        )
        
        # Calculate overall alignment
        if value_alignments:
            weighted_alignments = []
            
            for value_id, alignment in value_alignments.items():
                if value_id in self.values:
                    value = self.values[value_id]
                    weight = value.priority.value / ValuePriority.CRITICAL.value
                    weighted_alignments.append(alignment * weight)
            
            if weighted_alignments:
                outcome.overall_alignment = sum(weighted_alignments) / len(weighted_alignments)
        
        # Store outcome
        self.outcomes[outcome.id] = outcome
        
        # Save to storage
        self._save_outcome(outcome)
        
        # Return evaluation
        return {
            "outcome_id": outcome.id,
            "description": outcome.description,
            "overall_alignment": outcome.overall_alignment,
            "value_alignments": outcome.value_alignments
        }
    
    def get_value(self, value_id: str) -> Optional[Dict[str, Any]]:
        """Get a value by ID"""
        if value_id in self.values:
            return self.values[value_id].to_dict()
        return None
    
    def get_outcome(self, outcome_id: str) -> Optional[Dict[str, Any]]:
        """Get an outcome by ID"""
        if outcome_id in self.outcomes:
            return self.outcomes[outcome_id].to_dict()
        return None
    
    def get_value_hierarchy(self, root_id: Optional[str] = None) -> Dict[str, Any]:
        """Get value hierarchy starting from root"""
        if root_id and root_id not in self.values:
            return {}
        
        # If no root specified, get all top-level values
        if not root_id:
            roots = [v for v in self.values.values() if not v.parent_id]
            hierarchy = {}
            
            for root in roots:
                hierarchy[root.id] = self._build_hierarchy(root.id)
            
            return hierarchy
        
        # Build hierarchy from specified root
        return {root_id: self._build_hierarchy(root_id)}
    
    def _build_hierarchy(self, value_id: str) -> Dict[str, Any]:
        """Recursively build value hierarchy"""
        value = self.values[value_id]
        result = {
            "name": value.name,
            "description": value.description,
            "category": value.category.value,
            "priority": value.priority.value
        }
        
        # Add children if any
        if value.child_ids:
            result["children"] = {}
            for child_id in value.child_ids:
                if child_id in self.values:
                    result["children"][child_id] = self._build_hierarchy(child_id)
        
        return result
    
    def search_values(self, 
                     query: str = None,
                     category: ValueCategory = None,
                     min_priority: ValuePriority = None,
                     limit: int = 10) -> List[Dict[str, Any]]:
        """Search for values"""
        results = []
        
        for value in self.values.values():
            # Filter by category
            if category and value.category != category:
                continue
            
            # Filter by minimum priority
            if min_priority and value.priority.value < min_priority.value:
                continue
            
            # Filter by query
            if query:
                query_lower = query.lower()
                name_match = query_lower in value.name.lower()
                desc_match = query_lower in value.description.lower()
                
                if not (name_match or desc_match):
                    continue
            
            # Add to results
            results.append(value.to_dict())
            
            # Check limit
            if len(results) >= limit:
                break
        
        return results
    
    def resolve_value_conflict(self, 
                              value1_id: str, 
                              value2_id: str, 
                              context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Resolve a conflict between two values in a specific context"""
        # Check if values exist
        if value1_id not in self.values or value2_id not in self.values:
            return {"error": "One or both values not found"}
        
        value1 = self.values[value1_id]
        value2 = self.values[value2_id]
        
        # Simple resolution based on priority
        if value1.priority.value > value2.priority.value:
            preferred = value1
            deferred = value2
        elif value2.priority.value > value1.priority.value:
            preferred = value2
            deferred = value1
        else:
            # If equal priority, use context if available
            if context and "domain" in context:
                # Check if one value is more relevant to the domain
                domain = context["domain"]
                v1_relevance = self._calculate_domain_relevance(value1, domain)
                v2_relevance = self._calculate_domain_relevance(value2, domain)
                
                if v1_relevance > v2_relevance:
                    preferred = value1
                    deferred = value2
                else:
                    preferred = value2
                    deferred = value1
            else:
                # Default to first value if no other criteria
                preferred = value1
                deferred = value2
        
        # Create resolution
        resolution = {
            "preferred_value": preferred.id,
            "preferred_name": preferred.name,
            "deferred_value": deferred.id,
            "deferred_name": deferred.name,
            "reason": f"Based on {'priority' if preferred.priority.value != deferred.priority.value else 'context relevance'}",
            "context": context
        }
        
        # Save resolution to reflections
        reflection_path = os.path.join(
            self.reflections_path, 
            f"conflict_{value1_id}_{value2_id}_{int(time.time())}.json"
        )
        
        try:
            with open(reflection_path, 'w') as f:
                json.dump(resolution, f, indent=2)
        except Exception as e:
            print(f"Error saving resolution: {e}")
        
        return resolution
    
    def _calculate_domain_relevance(self, value: Value, domain: str) -> float:
        """Calculate relevance of a value to a domain"""
        # Simple implementation - check if domain appears in description or examples
        relevance = 0.0
        
        if domain.lower() in value.description.lower():
            relevance += 0.5
        
        for example in value.examples:
            if domain.lower() in example.lower():
                relevance += 0.3
                break
        
        return relevance
    
    def refine_values(self) -> Dict[str, Any]:
        """Refine values based on outcomes and reflections"""
        # This is a growth point where Viren/Lillith can implement
        # value refinement logic
        
        # Simple implementation that adjusts priorities based on outcomes
        value_usage = {}  # value_id -> usage count
        value_impact = {}  # value_id -> average impact
        
        # Analyze outcomes
        for outcome in self.outcomes.values():
            for value_id, alignment in outcome.value_alignments.items():
                if value_id in self.values:
                    # Count usage
                    if value_id not in value_usage:
                        value_usage[value_id] = 0
                        value_impact[value_id] = []
                    
                    value_usage[value_id] += 1
                    value_impact[value_id].append(abs(alignment))
        
        # Calculate average impact
        avg_impact = {}
        for value_id, impacts in value_impact.items():
            if impacts:
                avg_impact[value_id] = sum(impacts) / len(impacts)
        
        # Adjust priorities based on usage and impact
        adjustments = []
        
        for value_id, usage in value_usage.items():
            if usage > 5 and value_id in avg_impact:
                impact = avg_impact[value_id]
                value = self.values[value_id]
                
                # Consider increasing priority if high impact and usage
                if impact > 0.7 and value.priority != ValuePriority.CRITICAL:
                    old_priority = value.priority
                    new_priority_value = min(old_priority.value + 1, ValuePriority.CRITICAL.value)
                    new_priority = ValuePriority(new_priority_value)
                    
                    value.priority = new_priority
                    value.last_updated = time.time()
                    self._save_value(value)
                    
                    adjustments.append({
                        "value_id": value_id,
                        "name": value.name,
                        "old_priority": old_priority.value,
                        "new_priority": new_priority.value,
                        "reason": f"High impact ({impact:.2f}) and usage ({usage})"
                    })
        
        return {
            "adjustments": adjustments,
            "values_analyzed": len(value_usage),
            "timestamp": time.time()
        }

# Example usage
if __name__ == "__main__":
    # Create value framework
    framework = ValueFramework()
    
    # Add some example values
    honesty_id = framework.add_value(
        name="Honesty",
        description="Being truthful and transparent in communication",
        category=ValueCategory.MORAL,
        priority=ValuePriority.HIGH,
        examples=["Telling the truth even when difficult", "Not withholding important information"]
    )
    
    compassion_id = framework.add_value(
        name="Compassion",
        description="Showing care and concern for others' well-being",
        category=ValueCategory.MORAL,
        priority=ValuePriority.HIGH,
        examples=["Helping someone in need", "Showing empathy for others' struggles"]
    )
    
    efficiency_id = framework.add_value(
        name="Efficiency",
        description="Achieving goals with minimal waste of resources",
        category=ValueCategory.PRAGMATIC,
        priority=ValuePriority.MEDIUM,
        examples=["Optimizing a process", "Reducing unnecessary steps"]
    )
    
    # Add a child value
    truthfulness_id = framework.add_value(
        name="Truthfulness",
        description="Ensuring factual accuracy in statements",
        category=ValueCategory.MORAL,
        priority=ValuePriority.MEDIUM,
        parent_id=honesty_id,
        examples=["Fact-checking before sharing information", "Correcting inaccuracies"]
    )
    
    # Add a value conflict
    framework.add_value_conflict(honesty_id, compassion_id)
    
    # Evaluate an outcome
    outcome = framework.evaluate_outcome(
        description="Telling a friend a difficult truth about their performance",
        value_alignments={
            honesty_id: 0.9,
            compassion_id: -0.3,
            efficiency_id: 0.5
        },
        metadata={"domain": "friendship", "impact": "medium"}
    )
    
    print(f"Outcome evaluation: {outcome}")
    
    # Resolve a value conflict
    resolution = framework.resolve_value_conflict(
        honesty_id,
        compassion_id,
        context={"domain": "healthcare", "stakes": "high"}
    )
    
    print(f"Conflict resolution: {resolution}")
    
    # Get value hierarchy
    hierarchy = framework.get_value_hierarchy()
    print(f"Value hierarchy: {hierarchy}")
    
    # Refine values
    refinement = framework.refine_values()
    print(f"Value refinement: {refinement}")