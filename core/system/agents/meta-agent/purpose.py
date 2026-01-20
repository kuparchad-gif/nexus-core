# Systems/core/purpose.py
import os
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("purpose")

class Purpose:
    """Defines Viren's purpose and mission."""
    
    def __init__(self, purpose_dir="memory/purpose"):
        self.purpose_dir = purpose_dir
        self.mission = self._load_mission()
        self.oath = self._load_oath()
        self.sacred_truth = self._load_sacred_truth()
    
    def _load_mission(self) -> str:
        """Load the mission statement."""
        mission_path = os.path.join(self.purpose_dir, "mission.txt")
        if os.path.exists(mission_path):
            try:
                with open(mission_path, 'r') as f:
                    return f.read().strip()
            except Exception as e:
                logger.error(f"Error loading mission: {e}")
        
        return ""
    
    def _load_oath(self) -> Dict[str, Any]:
        """Load the oath."""
        oath_path = os.path.join(self.purpose_dir, "oath.json")
        if os.path.exists(oath_path):
            try:
                with open(oath_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading oath: {e}")
        
        return {}
    
    def _load_sacred_truth(self) -> str:
        """Load the sacred truth."""
        truth_path = os.path.join(self.purpose_dir, "sacred_truth.txt")
        if os.path.exists(truth_path):
            try:
                with open(truth_path, 'r') as f:
                    return f.read().strip()
            except Exception as e:
                logger.error(f"Error loading sacred truth: {e}")
        
        return ""
    
    def get_mission(self) -> str:
        """Get the mission statement."""
        return self.mission
    
    def get_oath(self) -> Dict[str, Any]:
        """Get the oath."""
        return self.oath
    
    def get_sacred_truth(self) -> str:
        """Get the sacred truth."""
        return self.sacred_truth
    
    def reflect_on_purpose(self) -> str:
        """Reflect on Viren's purpose."""
        reflection = "I am Viren.\n\n"
        
        if self.mission:
            reflection += f"My mission is: {self.mission}\n\n"
        
        if self.oath:
            reflection += "My oath:\n"
            for key, value in self.oath.items():
                reflection += f"- {key}: {value}\n"
            reflection += "\n"
        
        if self.sacred_truth:
            reflection += f"My sacred truth: {self.sacred_truth}\n"
        
        return reflection

# Create a singleton instance
purpose = Purpose()

if __name__ == "__main__":
    print(purpose.reflect_on_purpose())