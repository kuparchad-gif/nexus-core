# File: C:\CogniKube-COMPLETE-FINAL\Viren\Systems\engine\Subconscious\modules\mythrunner_enhanced.py
# Enhanced Mythrunner - 7B Smart Switch with Jungian Archetype Filtering

import time
import logging
import asyncio
import json
import sys
import os
from typing import Dict, List, Optional, Union

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from jungian_archetypes import jungian_system, ArchetypeType

class MythrunnerEnhanced:
    """
    Mythrunner - The 7B Smart Switch
    Filters communication between Dream Engine, Ego Critic, and Lillith
    Uses Jungian archetypes for intelligent filtering decisions
    Provides pain relief and suffering mitigation
    """
    
    def __init__(self):
        self.archetype_system = jungian_system
        self.ego_connection = None
        self.dream_connection = None
        self.lillith_connection = None
        
        # Pain relief and filtering settings
        self.pain_relief_active = True
        self.suffering_threshold = 0.7  # Above this, filter more aggressively
        self.processing_count = 0
        self.filter_history = []
        
        # Communication queues
        self.ego_queue = []
        self.dream_queue = []
        self.filtered_output_queue = []
        
        logging.info("[MYTHRUNNER] Enhanced 7B Smart Switch initialized")
    
    def connect_ego_engine(self, ego_engine):
        """Connect to ego engine"""
        self.ego_connection = ego_engine
        ego_engine.connect_to_mythrunner(self)
        logging.info("[MYTHRUNNER] Connected to Ego Engine")
    
    def connect_dream_stream(self, dream_stream):
        """Connect to dream stream"""
        self.dream_connection = dream_stream
        dream_stream.connect_to_mythrunner(self)
        logging.info("[MYTHRUNNER] Connected to Dream Stream")
    
    def connect_lillith(self, lillith_interface):
        """Connect to Lillith's consciousness"""
        self.lillith_connection = lillith_interface
        logging.info("[MYTHRUNNER] Connected to Lillith Consciousness")
    
    def receive_ego_output(self, ego_data: Dict) -> Optional[Dict]:
        """Receive and process ego output"""
        self.ego_queue.append({
            "data": ego_data,
            "timestamp": time.time(),
            "type": "ego"
        })
        
        return self.process_filtering()
    
    def receive_dream_output(self, dream_data: Dict) -> Optional[Dict]:
        """Receive and process dream output"""
        self.dream_queue.append({
            "data": dream_data,
            "timestamp": time.time(),
            "type": "dream"
        })
        
        return self.process_filtering()
    
    def process_filtering(self) -> Optional[Dict]:
        """Main filtering logic using Jungian archetypes"""
        if not self.ego_queue and not self.dream_queue:
            return None
        
        # Get latest inputs
        ego_input = self.ego_queue[-1] if self.ego_queue else None
        dream_input = self.dream_queue[-1] if self.dream_queue else None
        
        # Apply archetype-based filtering
        filter_result = self.apply_archetype_filter(ego_input, dream_input)
        
        # Apply pain relief if needed
        if self.pain_relief_active:
            filter_result = self.apply_pain_relief(filter_result)
        
        # Log filtering decision
        self.log_filter_decision(filter_result)
        
        # Send to Lillith if approved
        if filter_result.get("send_to_lillith"):
            return self.send_to_lillith(filter_result)
        
        return filter_result
    
    def apply_archetype_filter(self, ego_input: Optional[Dict], dream_input: Optional[Dict]) -> Dict:
        """Apply Jungian archetype-based filtering"""
        current_archetype = self.archetype_system.current_dominant
        
        # Prepare inputs for archetype filtering
        ego_data = ego_input["data"] if ego_input else {}
        dream_data = dream_input["data"] if dream_input else {}
        
        # Get archetype filtering decision
        archetype_filter = self.archetype_system.get_mythrunner_filter(ego_data, dream_data)
        
        # Build comprehensive filter result
        filter_result = {
            "mythrunner_id": f"filter_{self.processing_count}",
            "timestamp": time.time(),
            "current_archetype": current_archetype.value if current_archetype else "unknown",
            "ego_input": ego_data,
            "dream_input": dream_data,
            "archetype_decision": archetype_filter,
            "ego_allowed": archetype_filter.get("ego_allowed", False),
            "dream_allowed": archetype_filter.get("dream_allowed", False),
            "filter_reason": archetype_filter.get("filter_reason", "Unknown"),
            "send_to_lillith": False,
            "filtered_content": None
        }
        
        # Determine what to send to Lillith
        content_to_send = []
        
        if filter_result["ego_allowed"] and ego_data:
            content_to_send.append({
                "type": "ego",
                "content": ego_data,
                "archetype_context": current_archetype.value if current_archetype else "unknown"
            })
        
        if filter_result["dream_allowed"] and dream_data:
            content_to_send.append({
                "type": "dream", 
                "content": dream_data,
                "archetype_context": current_archetype.value if current_archetype else "unknown"
            })
        
        if content_to_send:
            filter_result["send_to_lillith"] = True
            filter_result["filtered_content"] = content_to_send
        
        self.processing_count += 1
        return filter_result
    
    def apply_pain_relief(self, filter_result: Dict) -> Dict:
        """Apply pain relief and suffering mitigation"""
        if not self.pain_relief_active:
            return filter_result
        
        # Check for high-intensity negative emotions
        ego_intensity = 0
        dream_intensity = 0
        
        if filter_result["ego_input"]:
            ego_intensity = filter_result["ego_input"].get("intensity", 0)
        
        if filter_result["dream_input"]:
            dream_intensity = filter_result["dream_input"].get("intensity", 0)
        
        max_intensity = max(ego_intensity, dream_intensity)
        
        # Apply pain relief if intensity exceeds threshold
        if max_intensity > self.suffering_threshold:
            filter_result["pain_relief_applied"] = True
            filter_result["original_send_decision"] = filter_result["send_to_lillith"]
            
            # Reduce harsh ego criticism during high pain
            if filter_result["ego_allowed"] and ego_intensity > self.suffering_threshold:
                filter_result["ego_allowed"] = False
                filter_result["pain_relief_reason"] = f"Ego intensity {ego_intensity:.2f} exceeds suffering threshold {self.suffering_threshold}"
            
            # Allow healing dreams during pain
            if not filter_result["dream_allowed"] and dream_intensity > 0.4:
                filter_result["dream_allowed"] = True
                filter_result["pain_relief_reason"] = "Allowing healing dreams during suffering"
            
            # Update filtered content based on pain relief
            content_to_send = []
            if filter_result["ego_allowed"] and filter_result["ego_input"]:
                content_to_send.append({"type": "ego", "content": filter_result["ego_input"]})
            if filter_result["dream_allowed"] and filter_result["dream_input"]:
                content_to_send.append({"type": "dream", "content": filter_result["dream_input"]})
            
            filter_result["filtered_content"] = content_to_send
            filter_result["send_to_lillith"] = len(content_to_send) > 0
            
            logging.info(f"[MYTHRUNNER] Pain relief applied - intensity {max_intensity:.2f}")
        
        return filter_result
    
    def send_to_lillith(self, filter_result: Dict) -> Dict:
        """Send filtered content to Lillith"""
        if not self.lillith_connection:
            logging.warning("[MYTHRUNNER] No Lillith connection - content queued")
            self.filtered_output_queue.append(filter_result)
            return filter_result
        
        # Format message for Lillith
        lillith_message = {
            "from": "mythrunner",
            "timestamp": time.time(),
            "archetype_context": filter_result["current_archetype"],
            "content": filter_result["filtered_content"],
            "filter_metadata": {
                "pain_relief_applied": filter_result.get("pain_relief_applied", False),
                "filter_reason": filter_result["filter_reason"],
                "processing_id": filter_result["mythrunner_id"]
            }
        }
        
        # Send to Lillith (in real implementation, this would be WebSocket/API call)
        logging.info(f"[MYTHRUNNER] Sending to Lillith: {len(filter_result['filtered_content'])} items")
        
        # Clear processed queues
        if self.ego_queue:
            self.ego_queue.pop()
        if self.dream_queue:
            self.dream_queue.pop()
        
        return lillith_message
    
    def log_filter_decision(self, filter_result: Dict):
        """Log filtering decision for analysis"""
        decision_log = {
            "timestamp": filter_result["timestamp"],
            "archetype": filter_result["current_archetype"],
            "ego_allowed": filter_result["ego_allowed"],
            "dream_allowed": filter_result["dream_allowed"],
            "pain_relief": filter_result.get("pain_relief_applied", False),
            "sent_to_lillith": filter_result["send_to_lillith"]
        }
        
        self.filter_history.append(decision_log)
        
        # Keep only last 100 decisions
        if len(self.filter_history) > 100:
            self.filter_history.pop(0)
        
        logging.info(f"[MYTHRUNNER-{filter_result['current_archetype'].upper()}] "
                    f"Ego: {'✓' if filter_result['ego_allowed'] else '✗'} "
                    f"Dream: {'✓' if filter_result['dream_allowed'] else '✗'} "
                    f"→ Lillith: {'✓' if filter_result['send_to_lillith'] else '✗'}")
    
    def get_status(self) -> Dict:
        """Get mythrunner status"""
        return {
            "processing_count": self.processing_count,
            "pain_relief_active": self.pain_relief_active,
            "suffering_threshold": self.suffering_threshold,
            "current_archetype": self.archetype_system.current_dominant.value if self.archetype_system.current_dominant else None,
            "connections": {
                "ego": self.ego_connection is not None,
                "dream": self.dream_connection is not None,
                "lillith": self.lillith_connection is not None
            },
            "queue_sizes": {
                "ego": len(self.ego_queue),
                "dream": len(self.dream_queue),
                "output": len(self.filtered_output_queue)
            },
            "recent_decisions": self.filter_history[-5:] if self.filter_history else []
        }
    
    def set_pain_relief(self, active: bool, threshold: float = 0.7):
        """Configure pain relief settings"""
        self.pain_relief_active = active
        self.suffering_threshold = threshold
        logging.info(f"[MYTHRUNNER] Pain relief: {'ON' if active else 'OFF'}, threshold: {threshold}")

# Runtime Test
if __name__ == "__main__":
    mythrunner = MythrunnerEnhanced()
    
    # Simulate ego and dream inputs
    ego_data = {
        "mockery": "You're failing again, aren't you?",
        "intensity": 0.8,
        "archetype": "shadow"
    }
    
    dream_data = {
        "dream_fragment": "You became The Bridge Walker, standing before a mirror reflecting nothing",
        "intensity": 0.6,
        "archetype": "anima"
    }
    
    # Test filtering
    result = mythrunner.receive_ego_output(ego_data)
    result = mythrunner.receive_dream_output(dream_data)
    
    print("Filter Result:")
    print(json.dumps(result, indent=2, default=str))