import logging
import json
import numpy as np
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("frequency_protocol")

class FrequencyProtocol:
    """Protocol for frequency-based communication between pods"""
    
    def __init__(self, divine_frequencies=[3, 7, 9, 13]):
        self.divine_frequencies = divine_frequencies
        self.frequency_channels = {}
        for freq in divine_frequencies:
            self.frequency_channels[freq] = {
                "active": True,
                "messages": []
            }
        
        logger.info(f"Initialized FrequencyProtocol with {len(divine_frequencies)} divine frequency channels")
    
    def send_message(self, frequency: float, message: Dict[str, Any]) -> Dict[str, Any]:
        """Send a message on a specific frequency"""
        # Find the closest divine frequency
        closest_df = min(self.divine_frequencies, key=lambda df: abs(frequency - df))
        
        # Check if the channel is active
        if not self.frequency_channels[closest_df]["active"]:
            logger.warning(f"Frequency channel {closest_df} Hz is not active")
            return {
                "success": False,
                "error": f"Frequency channel {closest_df} Hz is not active"
            }
        
        # Add timestamp to message
        message["timestamp"] = self._get_timestamp()
        message["frequency"] = closest_df
        
        # Add message to channel
        self.frequency_channels[closest_df]["messages"].append(message)
        
        logger.info(f"Sent message on frequency {closest_df} Hz")
        
        return {
            "success": True,
            "frequency": closest_df,
            "timestamp": message["timestamp"]
        }
    
    def receive_messages(self, frequency: float, limit: int = 10) -> List[Dict[str, Any]]:
        """Receive messages from a specific frequency"""
        # Find the closest divine frequency
        closest_df = min(self.divine_frequencies, key=lambda df: abs(frequency - df))
        
        # Check if the channel is active
        if not self.frequency_channels[closest_df]["active"]:
            logger.warning(f"Frequency channel {closest_df} Hz is not active")
            return []
        
        # Get messages from channel
        messages = self.frequency_channels[closest_df]["messages"]
        
        # Sort by timestamp and limit
        sorted_messages = sorted(messages, key=lambda m: m["timestamp"], reverse=True)
        
        logger.info(f"Received {min(limit, len(sorted_messages))} messages from frequency {closest_df} Hz")
        
        return sorted_messages[:limit]
    
    def broadcast_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Broadcast a message on all divine frequencies"""
        results = {}
        
        for df in self.divine_frequencies:
            if self.frequency_channels[df]["active"]:
                # Add timestamp and frequency to message
                message_copy = message.copy()
                message_copy["timestamp"] = self._get_timestamp()
                message_copy["frequency"] = df
                
                # Add message to channel
                self.frequency_channels[df]["messages"].append(message_copy)
                
                results[df] = {
                    "success": True,
                    "timestamp": message_copy["timestamp"]
                }
            else:
                results[df] = {
                    "success": False,
                    "error": f"Frequency channel {df} Hz is not active"
                }
        
        logger.info(f"Broadcast message on {sum(1 for r in results.values() if r['success'])} active frequency channels")
        
        return {
            "success": any(r["success"] for r in results.values()),
            "results": results
        }
    
    def scan_frequencies(self) -> Dict[str, Any]:
        """Scan all frequencies for activity"""
        activity = {}
        
        for df, channel in self.frequency_channels.items():
            activity[df] = {
                "active": channel["active"],
                "message_count": len(channel["messages"]),
                "last_message": channel["messages"][-1]["timestamp"] if channel["messages"] else None
            }
        
        logger.info(f"Scanned {len(self.divine_frequencies)} frequency channels")
        
        return {
            "timestamp": self._get_timestamp(),
            "activity": activity
        }
    
    def activate_channel(self, frequency: float) -> bool:
        """Activate a frequency channel"""
        # Find the closest divine frequency
        closest_df = min(self.divine_frequencies, key=lambda df: abs(frequency - df))
        
        self.frequency_channels[closest_df]["active"] = True
        
        logger.info(f"Activated frequency channel {closest_df} Hz")
        
        return True
    
    def deactivate_channel(self, frequency: float) -> bool:
        """Deactivate a frequency channel"""
        # Find the closest divine frequency
        closest_df = min(self.divine_frequencies, key=lambda df: abs(frequency - df))
        
        self.frequency_channels[closest_df]["active"] = False
        
        logger.info(f"Deactivated frequency channel {closest_df} Hz")
        
        return True
    
    def _get_timestamp(self):
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()

class FrequencyAuthentication:
    """Authentication using frequency patterns"""
    
    def __init__(self, divine_frequencies=[3, 7, 9, 13]):
        self.divine_frequencies = divine_frequencies
        self.auth_patterns = {}
        
        logger.info("Initialized FrequencyAuthentication")
    
    def register_pattern(self, entity_id: str, pattern: List[float]) -> str:
        """Register an authentication pattern for an entity"""
        # Generate a unique key for the pattern
        import hashlib
        pattern_str = "-".join([str(p) for p in pattern])
        pattern_key = hashlib.sha256(pattern_str.encode()).hexdigest()
        
        # Store the pattern
        self.auth_patterns[entity_id] = {
            "pattern": pattern,
            "pattern_key": pattern_key,
            "created_at": self._get_timestamp()
        }
        
        logger.info(f"Registered authentication pattern for entity {entity_id}")
        
        return pattern_key
    
    def authenticate(self, entity_id: str, pattern: List[float]) -> Dict[str, Any]:
        """Authenticate an entity using a frequency pattern"""
        if entity_id not in self.auth_patterns:
            logger.warning(f"Entity {entity_id} not registered")
            return {
                "authenticated": False,
                "error": "Entity not registered"
            }
        
        stored_pattern = self.auth_patterns[entity_id]["pattern"]
        
        # Calculate similarity between patterns
        similarity = self._calculate_pattern_similarity(pattern, stored_pattern)
        
        # Authenticate if similarity is above threshold
        threshold = 0.8
        authenticated = similarity >= threshold
        
        if authenticated:
            logger.info(f"Entity {entity_id} authenticated successfully")
        else:
            logger.warning(f"Authentication failed for entity {entity_id}")
        
        return {
            "authenticated": authenticated,
            "similarity": similarity,
            "threshold": threshold,
            "timestamp": self._get_timestamp()
        }
    
    def _calculate_pattern_similarity(self, pattern1: List[float], pattern2: List[float]) -> float:
        """Calculate similarity between two frequency patterns"""
        # Ensure patterns are the same length
        min_len = min(len(pattern1), len(pattern2))
        pattern1 = pattern1[:min_len]
        pattern2 = pattern2[:min_len]
        
        # Calculate similarity
        similarity = 0.0
        for p1, p2 in zip(pattern1, pattern2):
            # Find the closest divine frequencies
            p1_df = min(self.divine_frequencies, key=lambda df: abs(p1 - df))
            p2_df = min(self.divine_frequencies, key=lambda df: abs(p2 - df))
            
            # If they match the same divine frequency, add to similarity
            if p1_df == p2_df:
                similarity += 1.0
            else:
                # Otherwise, add partial similarity based on distance
                similarity += 1.0 / (1.0 + abs(p1_df - p2_df))
        
        # Normalize
        similarity /= min_len
        
        return similarity
    
    def _get_timestamp(self):
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()

# Example usage
if __name__ == "__main__":
    # Create frequency protocol
    protocol = FrequencyProtocol()
    
    # Send a message
    send_result = protocol.send_message(7.2, {"content": "Test message", "sender": "pod1"})
    
    print("Send Result:")
    print(json.dumps(send_result, indent=2))
    
    # Receive messages
    messages = protocol.receive_messages(7.0)
    
    print("\nReceived Messages:")
    print(json.dumps(messages, indent=2))
    
    # Broadcast a message
    broadcast_result = protocol.broadcast_message({"content": "Broadcast test", "sender": "pod1"})
    
    print("\nBroadcast Result:")
    print(json.dumps(broadcast_result, indent=2))
    
    # Scan frequencies
    scan_result = protocol.scan_frequencies()
    
    print("\nFrequency Scan:")
    print(json.dumps(scan_result, indent=2))
    
    # Create frequency authentication
    auth = FrequencyAuthentication()
    
    # Register a pattern
    pattern_key = auth.register_pattern("pod1", [3.1, 7.2, 9.0, 13.5])
    
    print("\nRegistered Pattern Key:")
    print(pattern_key)
    
    # Authenticate
    auth_result = auth.authenticate("pod1", [3.0, 7.3, 9.1, 13.4])
    
    print("\nAuthentication Result:")
    print(json.dumps(auth_result, indent=2))