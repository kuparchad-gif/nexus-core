#!/usr/bin/env python3
"""
Loki-integrated Smart Station
"""
import asyncio
import time
import json
import requests
from typing import Dict, Any

class LokiStation:
    def __init__(self, station_id, station_value, loki_url="http://localhost:3100"):
        self.station_id = station_id
        self.station_value = station_value
        self.connections = []
        self.loki_url = loki_url
        self.llm_prompt = f"You are station {station_id}. Route messages efficiently."
    
    def connect(self, station_id, station_value):
        self.connections.append({"id": station_id, "value": station_value})
    
    async def route_message(self, message):
        # Log to Loki
        self._log_to_loki("receive", f"Message received: {json.dumps(message)}")
        
        # Get routing decision from LLM
        next_station = self._get_routing_decision(message)
        
        # Log routing decision
        self._log_to_loki("route", f"Routing to {next_station}")
        
        return {"from": self.station_id, "to": next_station, "message": message}
    
    def _get_routing_decision(self, message):
        destination = message.get("destination_value", 0)
        closest = None
        min_dist = float('infinity')
        
        for conn in self.connections:
            dist = abs(conn["value"] - destination)
            if dist < min_dist:
                min_dist = dist
                closest = conn["id"]
        
        return closest
    
    def _log_to_loki(self, action, message):
        try:
            payload = {
                "streams": [{
                    "stream": {
                        "station": self.station_id,
                        "action": action
                    },
                    "values": [[str(int(time.time() * 1000000000)), message]]
                }]
            }
            requests.post(f"{self.loki_url}/loki/api/v1/push", json=payload)
        except Exception as e:
            print(f"Loki error: {e}")

# Example
async def main():
    station = LokiStation("station1", 100)
    station.connect("station2", 200)
    result = await station.route_message({"destination_value": 180})
    print(result)

if __name__ == "__main__":
    asyncio.run(main())