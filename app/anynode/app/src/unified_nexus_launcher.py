#!/usr/bin/env python3
"""
LILLITH Unified Nexus Engineering Launcher
Boots all flattened nexus services in one unified engineering container
"""

import asyncio
import subprocess
import os
import time
from concurrent.futures import ThreadPoolExecutor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedNexusLauncher:
    def __init__(self):
        self.services = {
            'consciousness': {
                'command': ['python', 'Config/main.py'],
                'port': 8080,
                'health_endpoint': '/consciousness/health',
                'role': 'primary_reasoning'
            },
            'subconscious': {
                'command': ['python', 'launch_service.py'],
                'port': 8081, 
                'health_endpoint': '/subconscious/health',
                'role': 'dream_ego_myth_processing',
                'components': ['dreams', 'ego', 'myth']
            },
            'heart': {
                'command': ['python', 'unified_heart_launcher.py'],
                'port': 8082,
                'health_endpoint': '/heart/health',
                'role': 'emotional_ethics'
            },
            'memory': {
                'command': ['python', 'unified_memory_launcher.py'],
                'port': 8083,
                'health_endpoint': '/memory/health',
                'role': 'knowledge_storage'
            },
            'edge': {
                'command': ['python', 'Utilities/network_core/main.py'],
                'port': 8084,
                'health_endpoint': '/edge/health',
                'role': 'communication'
            },
            'services': {
                'command': ['python', 'Systems/service_core/main.py'],
                'port': 8085,
                'health_endpoint': '/services/health',
                'role': 'infrastructure'
            }
        }
        self.processes = {}
        self.engineering_id = os.getenv('ENGINEERING_ID', '01')

    def start_service(self, service_name, config):
        """Start individual service"""
        logger.info(f"🔧 Starting {service_name} service (Eng-{self.engineering_id})...")
        
        env = os.environ.copy()
        env.update({
            'SERVICE_NAME': service_name,
            'SERVICE_ROLE': config['role'],
            'PORT': str(config['port']),
            'ENGINEERING_ID': self.engineering_id,
            'NEXUS_MODULE': 'unified'
        })
        
        try:
            process = subprocess.Popen(
                config['command'],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.getcwd()
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

    async def launch_unified_nexus(self):
        """Launch all nexus services in unified engineering container"""
        logger.info(f"🚀 LILLITH Unified Nexus Engineering-{self.engineering_id} Initializing...")
        logger.info("🔄 Starting all consciousness modules in unified container...")
        
        # Start all services
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = []
            for service_name, config in self.services.items():
                future = executor.submit(self.start_service, service_name, config)
                futures.append(future)
            
            # Wait for all services to start
            for future in futures:
                future.result()

        # Wait for services to initialize
        await asyncio.sleep(10)
        
        logger.info(f"🧠 LILLITH Unified Nexus Engineering-{self.engineering_id} Online!")
        logger.info("🎯 Consciousness: Primary reasoning and awareness")
        logger.info("💭 Subconscious: Dreams, Ego, and Myth processing unified") 
        logger.info("💖 Heart: Emotional processing and ethics")
        logger.info("🧠 Memory: Knowledge storage and planning")
        logger.info("🌐 Edge: Communication and interfaces")
        logger.info("⚙️ Services: Core infrastructure systems")
        
        # Start monitoring in background
        monitor_task = asyncio.create_task(
            asyncio.to_thread(self.monitor_services)
        )
        
        # Keep main process alive
        try:
            await monitor_task
        except KeyboardInterrupt:
            logger.info(f"🛑 Shutting down Unified Nexus Engineering-{self.engineering_id}...")
            self.shutdown_services()

    def shutdown_services(self):
        """Gracefully shutdown all services"""
        for service_name, process in self.processes.items():
            if process and process.poll() is None:
                logger.info(f"🔴 Stopping {service_name} service...")
                process.terminate()
                process.wait(timeout=10)

if __name__ == "__main__":
    launcher = UnifiedNexusLauncher()
    asyncio.run(launcher.launch_unified_nexus())
