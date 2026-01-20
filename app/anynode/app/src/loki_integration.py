#!/usr/bin/env python3
"""
Loki Integration for Gabriel's Horn Network
"""
import requests
import json
import time
from typing import Dict, Any

class LokiObserver:
    def __init__(self, loki_url="http://localhost:3100"):
        self.loki_url = loki_url
        self.push_endpoint = f"{loki_url}/loki/api/v1/push"
        self.query_endpoint = f"{loki_url}/loki/api/v1/query_range"
    
    def log_event(self, labels: Dict[str, str], message: str):
        """Log an event to Loki"""
        timestamp = str(int(time.time() * 1000000000))  # nanoseconds
        
        payload = {
            "streams": [
                {
                    "stream": labels,
                    "values": [[timestamp, message]]
                }
            ]
        }
        
        try:
            requests.post(self.push_endpoint, json=payload)
        except Exception as e:
            print(f"Loki logging error: {e}")
    
    def query_logs(self, query_string: str, start_time=None, end_time=None, limit=100):
        """Query Loki logs"""
        if not start_time:
            start_time = int((time.time() - 3600) * 1000000000)  # 1 hour ago
        if not end_time:
            end_time = int(time.time() * 1000000000)
        
        params = {
            "query": query_string,
            "start": start_time,
            "end": end_time,
            "limit": limit
        }
        
        try:
            response = requests.get(self.query_endpoint, params=params)
            return response.json()
        except Exception as e:
            print(f"Loki query error: {e}")
            return {"data": {"result": []}}

# Global instance
loki = LokiObserver()

# Decorator for LLM stations
def loki_monitored(func):
    """Decorator to add Loki monitoring to station functions"""
    async def wrapper(self, *args, **kwargs):
        station_id = getattr(self, "station_id", "unknown")
        
        # Log function call
        loki.log_event(
            {"station": station_id, "action": "call", "function": func.__name__},
            f"Called {func.__name__} with args: {args}, kwargs: {kwargs}"
        )
        
        # Call original function
        try:
            result = await func(self, *args, **kwargs)
            
            # Log success
            loki.log_event(
                {"station": station_id, "action": "success", "function": func.__name__},
                f"Result: {json.dumps(result)}"
            )
            
            return result
        except Exception as e:
            # Log error
            loki.log_event(
                {"station": station_id, "action": "error", "function": func.__name__},
                f"Error: {str(e)}"
            )
            raise
    
    return wrapper

# Example usage
if __name__ == "__main__":
    # Example of logging
    loki.log_event(
        {"component": "gabriel_horn", "severity": "info"},
        "Gabriel's Horn network initialized"
    )
    
    # Example of querying
    results = loki.query_logs('{component="gabriel_horn"}')
    print(f"Query results: {results}")