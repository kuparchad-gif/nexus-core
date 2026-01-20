# white_rabbit.py - Nexus deployment and monitoring agent
import os
import sys
import time
import json
import logging
import requests
from datetime import datetime
import threading
import socket

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("WhiteRabbit")

# Constants
PORT = int(os.environ.get("PORT", 5004))
BLUEPRINT_STORE = os.environ.get("BLUEPRINT_STORE", "firestore")
ALERT_TOPIC = os.environ.get("ALERT_TOPIC", "lillith-repair-alerts")
LOG_BUCKET = os.environ.get("LOG_BUCKET", "lillith-repair-logs")
SERVICES = [
    {"name": "swarm_manager", "port": 8000},
    {"name": "llm_loader", "port": 8001},
    {"name": "viren_core", "port": 8004},
    {"name": "it-pro-diag", "port": 5003}
]

class WhiteRabbit:
    def __init__(self):
        self.hostname = socket.gethostname()
        self.ip_address = socket.gethostbyname(self.hostname)
        self.start_time = datetime.now()
        self.blueprints = {}
        self.service_status = {service["name"]: "unknown" for service in SERVICES}
        logger.info(f"White Rabbit initialized on {self.hostname} ({self.ip_address})")
        
    def check_service_health(self, service_name, port):
        """Check if a service is healthy by making a request to its health endpoint"""
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=5)
            if response.status_code == 200:
                return "healthy"
            return "unhealthy"
        except Exception as e:
            logger.warning(f"Health check failed for {service_name}: {str(e)}")
            return "unreachable"
    
    def monitor_services(self):
        """Monitor all services and update their status"""
        while True:
            for service in SERVICES:
                status = self.check_service_health(service["name"], service["port"])
                if self.service_status[service["name"]] != status:
                    logger.info(f"Service {service['name']} status changed: {self.service_status[service['name']]} -> {status}")
                    if status == "unhealthy" or (self.service_status[service["name"]] == "healthy" and status == "unreachable"):
                        self.send_alert(f"Service {service['name']} is {status}")
                self.service_status[service["name"]] = status
            
            # Log overall status every minute
            logger.info(f"Service status: {json.dumps(self.service_status)}")
            time.sleep(60)
    
    def register_blueprint(self):
        """Register this deployment's blueprint"""
        blueprint = {
            "hostname": self.hostname,
            "ip_address": self.ip_address,
            "deployment_time": self.start_time.isoformat(),
            "services": SERVICES,
            "environment": {
                "blueprint_store": BLUEPRINT_STORE,
                "alert_topic": ALERT_TOPIC,
                "log_bucket": LOG_BUCKET
            }
        }
        
        self.blueprints[self.hostname] = blueprint
        logger.info(f"Registered blueprint for {self.hostname}")
        
        # In a real implementation, this would store to Firestore
        # For now, we'll just log it
        logger.info(f"Blueprint: {json.dumps(blueprint)}")
        return blueprint
    
    def send_alert(self, message):
        """Send an alert to the configured alert topic"""
        alert = {
            "timestamp": datetime.now().isoformat(),
            "hostname": self.hostname,
            "message": message,
            "service_status": self.service_status
        }
        
        # In a real implementation, this would publish to Pub/Sub
        # For now, we'll just log it
        logger.warning(f"ALERT: {message}")
        logger.warning(f"Alert details: {json.dumps(alert)}")
        return alert
    
    def start_http_server(self):
        """Start a simple HTTP server for health checks and API endpoints"""
        from http.server import HTTPServer, BaseHTTPRequestHandler
        
        class WhiteRabbitHandler(BaseHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                self.white_rabbit = args[2]
                super().__init__(*args[:2], **kwargs)
            
            def do_GET(self):
                if self.path == "/health":
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "healthy"}).encode())
                elif self.path == "/status":
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        "hostname": self.white_rabbit.hostname,
                        "uptime": str(datetime.now() - self.white_rabbit.start_time),
                        "service_status": self.white_rabbit.service_status
                    }).encode())
                elif self.path == "/blueprint":
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(self.white_rabbit.blueprints.get(self.white_rabbit.hostname, {})).encode())
                else:
                    self.send_response(404)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Not found"}).encode())
        
        def handler(*args):
            WhiteRabbitHandler(*args, self)
        
        server = HTTPServer(("", PORT), handler)
        logger.info(f"Starting HTTP server on port {PORT}")
        server.serve_forever()
    
    def run(self):
        """Run the White Rabbit agent"""
        logger.info("Starting White Rabbit agent")
        
        # Register blueprint
        self.register_blueprint()
        
        # Start monitoring in a separate thread
        monitor_thread = threading.Thread(target=self.monitor_services)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Start HTTP server
        self.start_http_server()

if __name__ == "__main__":
    white_rabbit = WhiteRabbit()
    white_rabbit.run()