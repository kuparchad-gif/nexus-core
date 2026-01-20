#!/usr/bin/env python3
"""
Enhanced LLM Station for Gabriel's Horn Network
Integrates with real LLMs and Loki monitoring
"""
import asyncio
import json
import time
from typing import Dict, Any, List, Optional, Union
from llm_manager import LLMManager
from enhanced_loki_observer import loki, loki_monitored

class LLMStation:
    """
    LLM-powered station in the Gabriel's Horn network
    Uses real LLM for routing decisions and integrates with Loki
    """
    
    def __init__(self, 
                station_id: str, 
                station_value: int, 
                model_name: str = "gemma-2b",
                llm_manager: Optional[LLMManager] = None):
        self.station_id = station_id
        self.station_value = station_value
        self.model_name = model_name
        self.connections = []
        self.llm_manager = llm_manager or LLMManager()
        self.system_prompt = f"""You are station {station_id} with value {station_value}.
Your job is to route messages to their destination efficiently.
Consider numerical proximity, network topology, and message priority.
You must decide which connected station to forward messages to."""
        
        # Station metrics
        self.metrics = {
            "messages_received": 0,
            "messages_routed": 0,
            "messages_delivered": 0,
            "routing_errors": 0,
            "avg_processing_time_ms": 0
        }
        
        # Log station creation
        loki.log_event(
            {"component": "station", "station": station_id, "action": "create", "model": model_name},
            f"Created station {station_id} with value {station_value} using model {model_name}"
        )
    
    def connect(self, station_id: str, station_value: int):
        """Connect to another station"""
        self.connections.append({"id": station_id, "value": station_value})
        
        # Log connection
        loki.log_event(
            {"component": "station", "station": self.station_id, "action": "connect"},
            f"Connected to station {station_id} with value {station_value}"
        )
    
    @loki_monitored(loki, "station")
    async def route_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route a message using the LLM
        
        Args:
            message: Message to route
            
        Returns:
            Routing decision
        """
        start_time = time.time()
        self.metrics["messages_routed"] += 1
        
        # Get routing history from Loki
        routing_history = await self._get_routing_history(message)
        
        # Get routing decision from LLM
        try:
            next_station = await self.llm_manager.get_routing_decision(
                self.station_id,
                message,
                self.connections,
                routing_history
            )
            
            # Update metrics
            elapsed_ms = int((time.time() - start_time) * 1000)
            self.metrics["avg_processing_time_ms"] = (
                (self.metrics["avg_processing_time_ms"] * (self.metrics["messages_routed"] - 1) + elapsed_ms) / 
                self.metrics["messages_routed"]
            )
            
            return {
                "action": "route",
                "from": self.station_id,
                "to": next_station,
                "message": message,
                "processing_time_ms": elapsed_ms
            }
        except Exception as e:
            # Log error
            self.metrics["routing_errors"] += 1
            loki.log_event(
                {"component": "station", "station": self.station_id, "action": "error", "error_type": "routing"},
                f"Routing error: {str(e)}",
                level="error"
            )
            
            # Fallback to numerical proximity
            next_station = self._get_fallback_routing(message)
            
            return {
                "action": "route",
                "from": self.station_id,
                "to": next_station,
                "message": message,
                "error": str(e),
                "fallback": True
            }
    
    async def _get_routing_history(self, message: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get routing history for a message from Loki"""
        # Extract message ID or create one if not present
        message_id = message.get("id", f"{message.get('source_id', 'unknown')}-{int(time.time())}")
        
        # Query Loki for routing history
        query = f'{{component="station", action="route"}} |= "{message_id}"'
        results = loki.query_logs(query, limit=10)
        
        history = []
        for stream in results.get("data", {}).get("result", []):
            for _, log_line in stream.get("values", []):
                try:
                    # Extract routing info from log
                    if "Routing message" in log_line:
                        parts = log_line.split("Routing message")
                        if len(parts) > 1:
                            routing_data = json.loads(parts[1].strip())
                            history.append({
                                "from": routing_data.get("from"),
                                "to": routing_data.get("to"),
                                "timestamp": routing_data.get("timestamp")
                            })
                except:
                    continue
        
        return history
    
    def _get_fallback_routing(self, message: Dict[str, Any]) -> str:
        """Get fallback routing decision based on numerical proximity"""
        if not self.connections:
            return None
        
        destination_value = message.get("destination_value", 0)
        
        # Find closest station by value
        closest_station = None
        min_distance = float('infinity')
        
        for station in self.connections:
            distance = abs(station["value"] - destination_value)
            if distance < min_distance:
                min_distance = distance
                closest_station = station["id"]
        
        return closest_station
    
    @loki_monitored(loki, "station")
    async def process_incoming(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an incoming message
        
        Args:
            message: Incoming message
            
        Returns:
            Processing result
        """
        self.metrics["messages_received"] += 1
        
        # Log message receipt with custom labels
        loki.log_event(
            {
                "component": "station", 
                "station": self.station_id, 
                "action": "receive",
                "message_id": message.get("id", "unknown"),
                "source": message.get("source_id", "unknown"),
                "priority": message.get("priority", "normal")
            },
            f"Received message: {json.dumps(message)}"
        )
        
        # Check if this is the final destination
        if message.get("destination_value") == self.station_value:
            self.metrics["messages_delivered"] += 1
            
            # Log delivery
            loki.log_event(
                {
                    "component": "station", 
                    "station": self.station_id, 
                    "action": "deliver",
                    "message_id": message.get("id", "unknown"),
                    "source": message.get("source_id", "unknown"),
                    "hops": message.get("hops", 0)
                },
                f"Message delivered: {json.dumps(message)}"
            )
            
            # Process the message content with LLM if needed
            if "content" in message and isinstance(message["content"], str):
                response = await self.llm_manager.generate(
                    f"Process this message as station {self.station_id}: {message['content']}",
                    self.model_name,
                    max_tokens=100
                )
                
                processed_content = response.get("text", "Message processed")
            else:
                processed_content = "Message processed"
            
            return {
                "status": "delivered",
                "station": self.station_id,
                "response": processed_content,
                "timestamp": time.time()
            }
        
        # Otherwise, route to next station
        return await self.route_message(message)
    
    @loki_monitored(loki, "station")
    async def get_status(self) -> Dict[str, Any]:
        """
        Get station status
        
        Returns:
            Station status information
        """
        # Query Loki for station-specific metrics
        query = f'{{component="station", station="{self.station_id}"}}'
        results = loki.query_logs(query, limit=100)
        
        # Count log entries by action
        action_counts = {}
        for stream in results.get("data", {}).get("result", []):
            action = stream.get("stream", {}).get("action", "unknown")
            action_counts[action] = action_counts.get(action, 0) + len(stream.get("values", []))
        
        return {
            "station_id": self.station_id,
            "value": self.station_value,
            "model": self.model_name,
            "connections": len(self.connections),
            "connected_to": [conn["id"] for conn in self.connections],
            "metrics": self.metrics,
            "log_metrics": action_counts
        }

# Example usage
async def main():
    # Create LLM manager
    llm_manager = LLMManager()
    
    # Create stations
    station1 = LLMStation("horn1", 100, "gemma-2b", llm_manager)
    station2 = LLMStation("horn2", 200, "hermes-2-pro-llama-3-7b", llm_manager)
    station3 = LLMStation("pod1", 150, "gemma-2b", llm_manager)
    
    # Connect stations
    station1.connect("horn2", 200)
    station1.connect("pod1", 150)
    station2.connect("horn1", 100)
    station3.connect("horn1", 100)
    
    # Create a message
    message = {
        "id": "test-message-1",
        "content": "Test message content",
        "source_id": "pod1",
        "source_value": 150,
        "destination_value": 200,
        "priority": "high",
        "timestamp": time.time(),
        "hops": 0,
        "path": ["pod1"]
    }
    
    # Process message
    print("Processing message at pod1...")
    result1 = await station3.process_incoming(message)
    print(f"Result from pod1: {result1}")
    
    if result1.get("to") == "horn1":
        # Update message for next hop
        message["hops"] += 1
        message["path"].append("horn1")
        
        # Process at horn1
        print("Processing message at horn1...")
        result2 = await station1.process_incoming(message)
        print(f"Result from horn1: {result2}")
        
        if result2.get("to") == "horn2":
            # Update message for next hop
            message["hops"] += 1
            message["path"].append("horn2")
            
            # Process at horn2
            print("Processing message at horn2...")
            result3 = await station2.process_incoming(message)
            print(f"Result from horn2: {result3}")
    
    # Get station status
    status1 = await station1.get_status()
    print(f"Station horn1 status: {status1}")

if __name__ == "__main__":
    asyncio.run(main())