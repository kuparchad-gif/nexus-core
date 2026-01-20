#!/usr/bin/env python3
"""
LILLITH Heart Module - Unified Service Launcher
Boots Guardian, Trinity, and Pulse services together in one container
"""

import asyncio
import subprocess
import os
import time
from concurrent.futures import ThreadPoolExecutor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedHeartLauncher:
    def __init__(self):
        self.services = {
            'guardian': {
                'command': ['python', 'Utilities/guardian_core/main.py'],
                'port': 8080,
                'health_endpoint': '/health'
            },
            'trinity': {
                'command': ['python', 'app/main.py'],
                'port': 8081, 
                'health_endpoint': '/trinity/health'
            },
            'pulse': {
                'command': ['python', 'launch_service.py'],
                'port': 8082,
                'health_endpoint': '/pulse/health'
            }
        }
        self.processes = {}

    def start_service(self, service_name, config):
        """Start individual service"""
        logger.info(f"🔵 Starting {service_name} service...")
        
        env = os.environ.copy()
        env.update({
            'SERVICE_NAME': service_name,
            'PORT': str(config['port']),
            'HEART_MODULE': 'true'
        })
        
        try:
            process = subprocess.Popen(
                config['command'],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.processes[service_name] = process
            logger.info(f"✅ {service_name} service started on port {config['port']}")
            return process
        except Exception as e:
            logger.error(f"❌ Failed to start {service_name}: {e}")
            return None

    def monitor_services(self):
        """Monitor all services and restart if needed"""
        while True:
            for service_name, process in self.processes.items():
                if process.poll() is not None:  # Process has terminated
                    logger.warning(f"⚠️ {service_name} service terminated, restarting...")
                    config = self.services[service_name]
                    self.processes[service_name] = self.start_service(service_name, config)
            time.sleep(30)  # Check every 30 seconds

    async def launch_heart_module(self):
        """Launch all heart services together"""
        logger.info("💖 LILLITH Heart Module Initializing...")
        logger.info("🔄 Starting Guardian, Trinity, and Pulse services...")
        
        # Start all services
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for service_name, config in self.services.items():
                future = executor.submit(self.start_service, service_name, config)
                futures.append(future)
            
            # Wait for all services to start
            for future in futures:
                future.result()

        # Wait for services to initialize
        await asyncio.sleep(5)
        
        logger.info("💖 LILLITH Heart Module Online!")
        logger.info("🛡️ Guardian: Ethical oversight and protection")
        logger.info("🔱 Trinity: Emotional harmony and balance") 
        logger.info("💓 Pulse: Vital signs and system health")
        
        # Start monitoring in background
        monitor_task = asyncio.create_task(
            asyncio.to_thread(self.monitor_services)
        )
        
        # Keep main process alive
        try:
            await monitor_task
        except KeyboardInterrupt:
            logger.info("🛑 Shutting down Heart Module...")
            self.shutdown_services()

    def shutdown_services(self):
        """Gracefully shutdown all services"""
        for service_name, process in self.processes.items():
            if process and process.poll() is None:
                logger.info(f"🔴 Stopping {service_name} service...")
                process.terminate()
                process.wait(timeout=10)

if __name__ == "__main__":
    launcher = UnifiedHeartLauncher()
    asyncio.run(launcher.launch_heart_module())
