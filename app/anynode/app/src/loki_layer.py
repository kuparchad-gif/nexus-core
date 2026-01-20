#!/usr/bin/env python3
"""
CogniKube Loki Observability - Layer 4
Complete monitoring and audit logging
"""

import json
import time
import asyncio
import aiohttp
from auth_gateway import auth_gateway
from typing import Dict, Any

class LokiObserver:
    def __init__(self):
        self.loki_url = "http://localhost:3100"  # Loki endpoint
        self.auth_layer = auth_gateway
        self.metrics = {
            "requests_total": 0,
            "auth_failures": 0,
            "encryption_errors": 0,
            "binary_protocol_errors": 0
        }
        
    async def log_to_loki(self, level: str, message: str, labels: Dict[str, str]):
        """Send log to Loki"""
        timestamp = str(int(time.time() * 1000000000))  # Nanoseconds
        
        log_entry = {
            "streams": [{
                "stream": labels,
                "values": [[timestamp, message]]
            }]
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                await session.post(
                    f"{self.loki_url}/loki/api/v1/push",
                    json=log_entry,
                    headers={"Content-Type": "application/json"}
                )
        except Exception as e:
            print(f"Failed to send log to Loki: {e}")
    
    async def audit_request(self, client_ip: str, service: str, action: str, result: str, role: str = None):
        """Audit security-relevant requests"""
        labels = {
            "job": "cognikube",
            "service": service,
            "level": "audit",
            "client_ip": client_ip
        }
        
        message = json.dumps({
            "timestamp": time.time(),
            "client_ip": client_ip,
            "service": service,
            "action": action,
            "result": result,
            "role": role,
            "metrics": self.metrics.copy()
        })
        
        await self.log_to_loki("audit", message, labels)
    
    async def security_alert(self, alert_type: str, details: Dict[str, Any]):
        """Send security alert"""
        labels = {
            "job": "cognikube",
            "level": "alert",
            "alert_type": alert_type
        }
        
        message = json.dumps({
            "timestamp": time.time(),
            "alert_type": alert_type,
            "details": details,
            "severity": "high"
        })
        
        await self.log_to_loki("alert", message, labels)
    
    async def monitored_handler(self, client_socket, service: str):
        """Handle requests with complete monitoring"""
        client_ip = client_socket.getpeername()[0]
        
        try:
            # Log connection attempt
            await self.audit_request(client_ip, service, "connection", "started")
            self.metrics["requests_total"] += 1
            
            # Process through auth layer with monitoring
            await self.auth_layer.authenticated_handler(client_socket, service)
            
            # Log successful completion
            await self.audit_request(client_ip, service, "connection", "completed")
            
        except Exception as e:
            # Log error and send alert
            await self.audit_request(client_ip, service, "connection", f"error: {str(e)}")
            await self.security_alert("connection_error", {
                "client_ip": client_ip,
                "service": service,
                "error": str(e)
            })
            
            # Update metrics
            if "auth" in str(e).lower():
                self.metrics["auth_failures"] += 1
            elif "encrypt" in str(e).lower():
                self.metrics["encryption_errors"] += 1
            elif "binary" in str(e).lower():
                self.metrics["binary_protocol_errors"] += 1
        
        finally:
            client_socket.close()
    
    async def start_metrics_reporter(self):
        """Periodically report metrics to Loki"""
        while True:
            labels = {
                "job": "cognikube",
                "level": "metrics"
            }
            
            message = json.dumps({
                "timestamp": time.time(),
                "metrics": self.metrics.copy()
            })
            
            await self.log_to_loki("info", message, labels)
            await asyncio.sleep(60)  # Report every minute

# Loki observer instance
loki_observer = LokiObserver()