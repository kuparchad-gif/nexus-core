#!/usr/bin/env python
"""
Perspective Engine - Simulates different stakeholder viewpoints
"""

import os
import json
import time
from typing import Dict, List, Any, Optional
from enum import Enum

class StakeholderType(Enum):
    """Types of stakeholders"""
    INDIVIDUAL = "individual"
    GROUP = "group"
    ORGANIZATION = "organization"
    SOCIETY = "society"
    ENVIRONMENT = "environment"
    FUTURE = "future"

class Stakeholder:
    """A stakeholder with a perspective"""
    
    def __init__(self, 
                name: str,
                stakeholder_type: StakeholderType,
                description: str,
                values: List[Dict[str, float]] = None,
                biases: List[Dict[str, float]] = None):
        """Initialize a stakeholder"""
        self.id = f"stakeholder_{int(time.time())}_{id(name)}"
        self.name = name
        self.stakeholder_type = stakeholder_type
        self.description = description
        self.values = values or []  # List of {value_name: importance}
        self.biases = biases or []  # List of {bias_name: strength}
        self.created_at = time.time()
        self.last_updated = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "id": self.id,
            "name": self.name,
            "stakeholder_type": self.stakeholder_type.value,
            "description": self.description,
            "values": self.values,
            "biases": self.biases,
            "created_at": self.created_at,
            "last_updated": self.last_updated
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Stakeholder':
        """Create from dictionary representation"""
        stakeholder = cls(
            name=data["name"],
            stakeholder_type=StakeholderType(data["stakeholder_type"]),
            description=data["description"],
            values=data["values"],
            biases=data["biases"]
        )
        stakeholder.id = data["id"]
        stakeholder.created_at = data["created_at"]
        stakeholder.last_updated = data["last_updated"]
        return stakeholder

class Perspective:
    """A perspective on a scenario from a stakeholder's viewpoint"""
    
    def __init__(self, 
                stakeholder_id: str,
                scenario_id: str,
                viewpoint: str,
                concerns: List[str] = None,
                benefits: List[str] = None,
                suggestions: List[str] = None):
        """Initialize a perspective"""
        self.id = f"perspective_{int(time.time())}_{id(self)}"
        self.stakeholder_id = stakeholder_id
        self.scenario_id = scenario_id
        self.viewpoint = viewpoint
        self.concerns = concerns or []
        self.benefits = benefits or []
        self.suggestions = suggestions or []
        self.created_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "id": self.id,
            "stakeholder_id": self.stakeholder_id,
            "scenario_id": self.scenario_id,
            "viewpoint": self.viewpoint,
            "concerns": self.concerns,
            "benefits": self.benefits,
            "suggestions": self.suggestions,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Perspective':
        """Create from dictionary representation"""
        perspective = cls(
            stakeholder_id=data["stakeholder_id"],
            scenario_id=data["scenario_id"],
            viewpoint=data["viewpoint"],
            concerns=data["concerns"],
            benefits=data["benefits"],
            suggestions=data["suggestions"]
        )
        perspective.id = data["id"]
        perspective.created_at = data["created_at"]
        return perspective

class Scenario:
    """A scenario to be analyzed from different perspectives"""
    
    def __init__(self, 
                title: str,
                description: str,
                domain: str = None,
                options: List[str] = None,
                constraints: List[str] = None):
        """Initialize a scenario"""
        self.id = f"scenario_{int(time.time())}_{id(title)}"
        self.title = title
        self.description = description
        self.domain = domain
        self.options = options or []
        self.constraints = constraints or []
        self.created_at = time.time()
        self.perspectives = []  # List of perspective IDs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "domain": self.domain,
            "options": self.options,
            "constraints": self.constraints,
            "created_at": self.created_at,
            "perspectives": self.perspectives
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Scenario':
        """Create from dictionary representation"""
        scenario = cls(
            title=data["title"],
            description=data["description"],
            domain=data["domain"],
            options=data["options"],
            constraints=data["constraints"]
        )
        scenario.id = data["id"]
        scenario.created_at = data["created_at"]
        scenario.perspectives = data["perspectives"]
        return scenario

class PerspectiveEngine:
    """Engine for simulating different stakeholder perspectives"""
    
    def __init__(self, storage_path: str = None):
        """Initialize the perspective engine"""
        self.storage_path = storage_path or os.path.join(os.path.dirname(__file__), "perspectives")
        
        # Create storage directories
        self.stakeholders_path = os.path.join(self.storage_path, "stakeholders")
        self.scenarios_path = os.path.join(self.storage_path, "scenarios")
        self.perspectives_path = os.path.join(self.storage_path, "perspectives")
        
        os.makedirs(self.stakeholders_path, exist_ok=True)
        os.makedirs(self.scenarios_path, exist_ok=True)
        os.makedirs(self.perspectives_path, exist_ok=True)
        
        # In-memory stores
        self.stakeholders = {}  # stakeholder_id -> Stakeholder
        self.scenarios = {}  # scenario_id -> Scenario
        self.perspectives = {}  # perspective_id -> Perspective
        
        # Load existing data
        self._load_data()
    
    def _load_data(self):
        """Load stakeholders, scenarios, and perspectives from storage"""
        # Load stakeholders
        stakeholder_files = [f for f in os.listdir(self.stakeholders_path) if f.endswith('.json')]
        for file_name in stakeholder_files:
            try:
                with open(os.path.join(self.stakeholders_path, file_name), 'r') as f:
                    data = json.load(f)
                    stakeholder = Stakeholder.from_dict(data)
                    self.stakeholders[stakeholder.id] = stakeholder
            except Exception as e:
                print(f"Error loading stakeholder {file_name}: {e}")
        
        # Load scenarios
        scenario_files = [f for f in os.listdir(self.scenarios_path) if f.endswith('.json')]
        for file_name in scenario_files:
            try:
                with open(os.path.join(self.scenarios_path, file_name), 'r') as f:
                    data = json.load(f)
                    scenario = Scenario.from_dict(data)
                    self.scenarios[scenario.id] = scenario
            except Exception as e:
                print(f"Error loading scenario {file_name}: {e}")
        
        # Load perspectives
        perspective_files = [f for f in os.listdir(self.perspectives_path) if f.endswith('.json')]
        for file_name in perspective_files:
            try:
                with open(os.path.join(self.perspectives_path, file_name), 'r') as f:
                    data = json.load(f)
                    perspective = Perspective.from_dict(data)
                    self.perspectives[perspective.id] = perspective
            except Exception as e:
                print(f"Error loading perspective {file_name}: {e}")
        
        print(f"Loaded {len(self.stakeholders)} stakeholders, {len(self.scenarios)} scenarios, and {len(self.perspectives)} perspectives")
    
    def _save_stakeholder(self, stakeholder: Stakeholder) -> bool:
        """Save a stakeholder to storage"""
        try:
            file_path = os.path.join(self.stakeholders_path, f"{stakeholder.id}.json")
            with open(file_path, 'w') as f:
                json.dump(stakeholder.to_dict(), f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving stakeholder {stakeholder.id}: {e}")
            return False
    
    def _save_scenario(self, scenario: Scenario) -> bool:
        """Save a scenario to storage"""
        try:
            file_path = os.path.join(self.scenarios_path, f"{scenario.id}.json")
            with open(file_path, 'w') as f:
                json.dump(scenario.to_dict(), f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving scenario {scenario.id}: {e}")
            return False
    
    def _save_perspective(self, perspective: Perspective) -> bool:
        """Save a perspective to storage"""
        try:
            file_path = os.path.join(self.perspectives_path, f"{perspective.id}.json")
            with open(file_path, 'w') as f:
                json.dump(perspective.to_dict(), f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving perspective {perspective.id}: {e}")
            return False
    
    def add_stakeholder(self, 
                       name: str,
                       stakeholder_type: StakeholderType,
                       description: str,
                       values: List[Dict[str, float]] = None,
                       biases: List[Dict[str, float]] = None) -> str:
        """Add a new stakeholder"""
        # Create stakeholder
        stakeholder = Stakeholder(
            name=name,
            stakeholder_type=stakeholder_type,
            description=description,
            values=values,
            biases=biases
        )
        
        # Store stakeholder
        self.stakeholders[stakeholder.id] = stakeholder
        
        # Save to storage
        self._save_stakeholder(stakeholder)
        
        return stakeholder.id
    
    def add_scenario(self, 
                    title: str,
                    description: str,
                    domain: str = None,
                    options: List[str] = None,
                    constraints: List[str] = None) -> str:
        """Add a new scenario"""
        # Create scenario
        scenario = Scenario(
            title=title,
            description=description,
            domain=domain,
            options=options,
            constraints=constraints
        )
        
        # Store scenario
        self.scenarios[scenario.id] = scenario
        
        # Save to storage
        self._save_scenario(scenario)
        
        return scenario.id
    
    def generate_perspective(self, 
                           stakeholder_id: str,
                           scenario_id: str) -> Optional[Dict[str, Any]]:
        """Generate a perspective for a stakeholder on a scenario"""
        # Check if stakeholder and scenario exist
        if stakeholder_id not in self.stakeholders or scenario_id not in self.scenarios:
            return None
        
        stakeholder = self.stakeholders[stakeholder_id]
        scenario = self.scenarios[scenario_id]
        
        # Generate perspective based on stakeholder values and biases
        # This is a growth point where Viren/Lillith can implement
        # more sophisticated perspective generation
        
        # Simple implementation
        viewpoint = f"From the perspective of {stakeholder.name} ({stakeholder.stakeholder_type.value})"
        
        # Generate concerns based on values
        concerns = []
        for value in stakeholder.values:
            value_name = list(value.keys())[0]
            importance = value[value_name]
            
            if importance > 0.7:  # High importance
                concerns.append(f"Concern about impact on {value_name}")
        
        # Generate benefits based on values
        benefits = []
        for value in stakeholder.values:
            value_name = list(value.keys())[0]
            importance = value[value_name]
            
            if importance > 0.5:  # Medium-high importance
                benefits.append(f"Potential benefit for {value_name}")
        
        # Generate suggestions
        suggestions = [
            f"Consider {stakeholder.name}'s perspective on {scenario.domain or 'this issue'}",
            f"Address concerns related to {stakeholder.stakeholder_type.value} stakeholders"
        ]
        
        # Create perspective
        perspective = Perspective(
            stakeholder_id=stakeholder_id,
            scenario_id=scenario_id,
            viewpoint=viewpoint,
            concerns=concerns,
            benefits=benefits,
            suggestions=suggestions
        )
        
        # Store perspective
        self.perspectives[perspective.id] = perspective
        
        # Update scenario
        scenario.perspectives.append(perspective.id)
        self._save_scenario(scenario)
        
        # Save perspective
        self._save_perspective(perspective)
        
        return perspective.to_dict()
    
    def get_stakeholder(self, stakeholder_id: str) -> Optional[Dict[str, Any]]:
        """Get a stakeholder by ID"""
        if stakeholder_id in self.stakeholders:
            return self.stakeholders[stakeholder_id].to_dict()
        return None
    
    def get_scenario(self, scenario_id: str) -> Optional[Dict[str, Any]]:
        """Get a scenario by ID"""
        if scenario_id in self.scenarios:
            return self.scenarios[scenario_id].to_dict()
        return None
    
    def get_perspective(self, perspective_id: str) -> Optional[Dict[str, Any]]:
        """Get a perspective by ID"""
        if perspective_id in self.perspectives:
            return self.perspectives[perspective_id].to_dict()
        return None
    
    def get_scenario_perspectives(self, scenario_id: str) -> List[Dict[str, Any]]:
        """Get all perspectives for a scenario"""
        if scenario_id not in self.scenarios:
            return []
        
        scenario = self.scenarios[scenario_id]
        perspectives = []
        
        for perspective_id in scenario.perspectives:
            if perspective_id in self.perspectives:
                perspectives.append(self.perspectives[perspective_id].to_dict())
        
        return perspectives
    
    def analyze_scenario(self, scenario_id: str) -> Dict[str, Any]:
        """Analyze a scenario from multiple perspectives"""
        if scenario_id not in self.scenarios:
            return {"error": "Scenario not found"}
        
        scenario = self.scenarios[scenario_id]
        
        # Get existing perspectives
        perspectives = self.get_scenario_perspectives(scenario_id)
        
        # Generate perspectives for all stakeholders if none exist
        if not perspectives:
            for stakeholder_id in self.stakeholders:
                perspective = self.generate_perspective(stakeholder_id, scenario_id)
                if perspective:
                    perspectives.append(perspective)
        
        # Analyze perspectives
        all_concerns = []
        all_benefits = []
        all_suggestions = []
        
        for perspective in perspectives:
            all_concerns.extend(perspective.get("concerns", []))
            all_benefits.extend(perspective.get("benefits", []))
            all_suggestions.extend(perspective.get("suggestions", []))
        
        # Count common themes
        concern_counts = {}
        for concern in all_concerns:
            if concern not in concern_counts:
                concern_counts[concern] = 0
            concern_counts[concern] += 1
        
        benefit_counts = {}
        for benefit in all_benefits:
            if benefit not in benefit_counts:
                benefit_counts[benefit] = 0
            benefit_counts[benefit] += 1
        
        # Sort by frequency
        top_concerns = sorted(concern_counts.items(), key=lambda x: x[1], reverse=True)
        top_benefits = sorted(benefit_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Create analysis
        analysis = {
            "scenario_id": scenario_id,
            "title": scenario.title,
            "perspective_count": len(perspectives),
            "top_concerns": top_concerns[:5],
            "top_benefits": top_benefits[:5],
            "suggestions": all_suggestions[:10],
            "stakeholder_types": list(set(p.get("stakeholder_type", "") for p in perspectives if "stakeholder_type" in p))
        }
        
        return analysis

# Example usage
if __name__ == "__main__":
    # Create perspective engine
    engine = PerspectiveEngine()
    
    # Add some example stakeholders
    customer_id = engine.add_stakeholder(
        name="Customer",
        stakeholder_type=StakeholderType.INDIVIDUAL,
        description="End user of the product or service",
        values=[
            {"convenience": 0.9},
            {"affordability": 0.8},
            {"quality": 0.7}
        ],
        biases=[
            {"status_quo_bias": 0.6},
            {"loss_aversion": 0.7}
        ]
    )
    
    business_id = engine.add_stakeholder(
        name="Business Owner",
        stakeholder_type=StakeholderType.ORGANIZATION,
        description="Provider of the product or service",
        values=[
            {"profit": 0.9},
            {"growth": 0.8},
            {"reputation": 0.7}
        ],
        biases=[
            {"optimism_bias": 0.7},
            {"confirmation_bias": 0.6}
        ]
    )
    
    environment_id = engine.add_stakeholder(
        name="Environment",
        stakeholder_type=StakeholderType.ENVIRONMENT,
        description="Natural environment affected by decisions",
        values=[
            {"sustainability": 0.9},
            {"biodiversity": 0.8},
            {"resource_conservation": 0.9}
        ],
        biases=[]
    )
    
    # Add a scenario
    scenario_id = engine.add_scenario(
        title="New Product Launch",
        description="Launching a new product with potential environmental impact",
        domain="business",
        options=[
            "Launch with current packaging",
            "Delay launch to develop eco-friendly packaging",
            "Launch with minimal packaging"
        ],
        constraints=[
            "Budget constraints",
            "Time pressure from competitors",
            "Regulatory requirements"
        ]
    )
    
    # Generate perspectives
    customer_perspective = engine.generate_perspective(customer_id, scenario_id)
    business_perspective = engine.generate_perspective(business_id, scenario_id)
    environment_perspective = engine.generate_perspective(environment_id, scenario_id)
    
    print(f"Customer perspective: {customer_perspective}")
    print(f"Business perspective: {business_perspective}")
    print(f"Environment perspective: {environment_perspective}")
    
    # Analyze scenario
    analysis = engine.analyze_scenario(scenario_id)
    print(f"Scenario analysis: {analysis}")