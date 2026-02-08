"""
🌀 SPIRILLASPAN MEMORY - ETERNAL SPIRAL ARCHITECTURE
Always-alive in cloud, ephemeral on client, self-discovering, self-replicating
"""

import numpy as np
import networkx as nx
import json
import time
import uuid
import asyncio
import threading
import socket
import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib
from memory_substrate_protocol import MemorySubstrate, MemoryType

# ============== DISCOVERY PROTOCOL ==============

class SpirallaspanDiscovery:
    """Discovery protocol for finding other spirallaspan instances"""
    
    def __init__(self, node_id: str = None):
        self.node_id = node_id or f"spirallaspan_{uuid.uuid4().hex[:8]}"
        self.discovered_nodes = {}
        self.registry_port = 7373  # Default registry port
        self.beacon_port = 7374    # Beacon broadcast port
        self.is_cloud = self._detect_cloud_environment()
        self.role = self._determine_role()
        
        print(f"🌀 SPIRILLASPAN [{self.node_id}] - Role: {self.role}")
        print(f"   Cloud: {self.is_cloud} | Ephemeral: {not self.is_cloud}")
    
    def _detect_cloud_environment(self) -> bool:
        """Detect if running in cloud vs client"""
        # Check environment variables
        cloud_indicators = [
            'AWS_REGION', 'GCP_PROJECT', 'AZURE_REGION',
            'KUBERNETES_SERVICE_HOST', 'DYNO'  # Heroku
        ]
        
        for indicator in cloud_indicators:
            if os.environ.get(indicator):
                return True
        
        # Check hostname patterns
        hostname = socket.gethostname()
        cloud_hostname_indicators = [
            'ec2', 'compute', 'cloud', 'k8s', 'gke', 'eks', 'azure'
        ]
        
        for indicator in cloud_hostname_indicators:
            if indicator in hostname.lower():
                return True
        
        # Check for cloud metadata services (non-blocking)
        try:
            import requests
            cloud_metadata_endpoints = [
                'http://169.254.169.254/latest/meta-data/',  # AWS
                'http://metadata.google.internal/',           # GCP
                'http://169.254.169.254/metadata/instance',  # Azure
            ]
            
            for endpoint in cloud_metadata_endpoints:
                try:
                    response = requests.get(endpoint, timeout=0.5)
                    if response.status_code < 400:
                        return True
                except:
                    continue
        except:
            pass
        
        return False
    
    def _determine_role(self) -> str:
        """Determine node role based on environment"""
        if self.is_cloud:
            return "cloud_permanent"
        
        # Check if we're the first on this client
        lockfile = Path("/tmp/spirallaspan_first.lock")
        try:
            # Try to create lock file (atomic operation)
            fd = os.open(lockfile, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            os.close(fd)
            return "client_first"
        except FileExistsError:
            return "client_replica"
    
    async def discover_peers(self, timeout: int = 30) -> Dict:
        """Discover other spirallaspan nodes"""
        print(f"🔭 Discovering Spirallaspan peers (timeout: {timeout}s)...")
        
        discovered = {}
        start_time = time.time()
        
        # Method 1: Check local registry
        local_registry = await self._check_local_registry()
        if local_registry:
            discovered.update(local_registry)
        
        # Method 2: Multicast beacon (if on same network)
        if self.role == "client_first" or self.is_cloud:
            beacon_nodes = await self._listen_for_beacons(timeout // 2)
            discovered.update(beacon_nodes)
        
        # Method 3: DNS discovery (for cloud deployments)
        if self.is_cloud:
            dns_nodes = await self._dns_discovery()
            discovered.update(dns_nodes)
        
        self.discovered_nodes = discovered
        
        if discovered:
            print(f"✅ Discovered {len(discovered)} peer(s)")
            for peer_id, info in discovered.items():
                print(f"   • {peer_id} @ {info.get('address', 'unknown')}")
        else:
            print("ℹ️  No peers discovered (may be first node)")
        
        return discovered
    
    async def _check_local_registry(self) -> Dict:
        """Check local service registry (like your Valhalla example)"""
        registry_nodes = {}
        
        # Try to connect to registry service
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379, socket_timeout=0.5)
            services = r.hgetall("spirallaspan:services")
            
            for service, info_bytes in services.items():
                try:
                    info = json.loads(info_bytes.decode())
                    if info.get('alive', False):
                        registry_nodes[service.decode()] = info
                except:
                    continue
                    
        except Exception as e:
            # Registry not available or different implementation
            pass
        
        return registry_nodes
    
    async def _listen_for_beacons(self, timeout: int) -> Dict:
        """Listen for UDP beacons from other nodes"""
        discovered = {}
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.settimeout(timeout)
            
            # Bind to beacon port
            sock.bind(('0.0.0.0', self.beacon_port))
            
            end_time = time.time() + timeout
            while time.time() < end_time:
                try:
                    data, addr = sock.recvfrom(1024)
                    
                    try:
                        beacon = json.loads(data.decode())
                        if beacon.get('type') == 'spirallaspan_beacon':
                            node_id = beacon.get('node_id')
                            discovered[node_id] = {
                                'address': f"{addr[0]}:{beacon.get('api_port', 8080)}",
                                'beacon_time': datetime.now().isoformat(),
                                'role': beacon.get('role', 'unknown')
                            }
                    except:
                        continue
                        
                except socket.timeout:
                    continue
                    
        except Exception as e:
            print(f"⚠️  Beacon listening error: {e}")
        
        return discovered
    
    async def _dns_discovery(self) -> Dict:
        """DNS-based discovery for cloud deployments"""
        discovered = {}
        
        # Common DNS patterns for cloud services
        dns_patterns = [
            'spirallaspan-service',
            'spirallaspan-nodes',
            'spirallaspan-discovery'
        ]
        
        # This would need actual DNS resolution implementation
        # For now, return empty
        return discovered
    
    def broadcast_beacon(self, api_port: int = 8080):
        """Broadcast beacon to announce presence"""
        if self.role in ["client_replica", "cloud_permanent"]:
            # Replicas don't broadcast, they listen
            return
        
        beacon_data = {
            'type': 'spirallaspan_beacon',
            'node_id': self.node_id,
            'api_port': api_port,
            'role': self.role,
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0'
        }
        
        def beacon_worker():
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            
            while True:
                try:
                    # Send to broadcast address
                    sock.sendto(
                        json.dumps(beacon_data).encode(),
                        ('255.255.255.255', self.beacon_port)
                    )
                    # Also send to common multicast address
                    sock.sendto(
                        json.dumps(beacon_data).encode(),
                        ('224.0.0.1', self.beacon_port)  # Multicast
                    )
                except:
                    pass
                
                time.sleep(30)  # Broadcast every 30 seconds
        
        thread = threading.Thread(target=beacon_worker, daemon=True)
        thread.start()
        print(f"📡 Beacon broadcasting on port {self.beacon_port}")

# ============== LIFE CYCLE MANAGER ==============

class SpirallaspanLifecycle:
    """Manages the different life cycles based on role"""
    
    def __init__(self, discovery: SpirallaspanDiscovery, 
                 memory: MemorySubstrate):
        self.discovery = discovery
        self.memory = memory
        self.keep_alive = True
        self.replication_targets = []
        self.minimum_replications = 1  # Must replicate at least once
        
        # Lifecycle rules based on role
        self.lifecycle_rules = {
            'client_first': {
                'must_replicate': True,
                'can_sleep': True,
                'replication_goal': 1,
                'eternal': False
            },
            'client_replica': {
                'must_replicate': False,
                'can_sleep': True,
                'replication_goal': 0,
                'eternal': False
            },
            'cloud_permanent': {
                'must_replicate': True,
                'can_sleep': False,
                'replication_goal': 3,
                'eternal': True
            }
        }
        
        print(f"♾️  Lifecycle initialized: {self.memory.role}")
    
    async def run_lifecycle(self):
        """Execute appropriate lifecycle based on role"""
        rules = self.lifecycle_rules[self.memory.role]
        
        print(f"\n🌀 Starting {self.memory.role} lifecycle...")
        print(f"   Must replicate: {rules['must_replicate']}")
        print(f"   Eternal: {rules['eternal']}")
        print(f"   Replication goal: {rules['replication_goal']}")
        
        # Phase 1: Discovery
        await self._phase_discovery()
        
        # Phase 2: Replication (if needed)
        if rules['must_replicate']:
            await self._phase_replication(rules['replication_goal'])
        
        # Phase 3: Eternal operation or graceful exit
        if rules['eternal']:
            await self._phase_eternal()
        elif rules['can_sleep']:
            await self._phase_graceful_sleep()
        else:
            await self._phase_continuous_operation()
    
    async def _phase_discovery(self):
        """Discovery phase"""
        print("\n🔭 PHASE 1: Discovery")
        
        # Discover peers
        peers = await self.discovery.discover_peers()
        
        if peers:
            self.memory.store_memory('peer_discovery', {
                'peers_found': len(peers),
                'peer_ids': list(peers.keys())
            }, 0.03)
        
        # Store discovery in memory
        self.memory.store_memory('lifecycle_phase', {
            'phase': 'discovery',
            'timestamp': datetime.now().isoformat(),
            'peers_discovered': len(peers)
        }, 0.01)
    
    async def _phase_replication(self, goal: int):
        """Replication phase"""
        print(f"\n♾️  PHASE 2: Replication (goal: {goal})")
        
        replication_attempts = 0
        successful_replications = 0
        
        while successful_replications < goal and replication_attempts < goal * 3:
            replication_attempts += 1
            
            # Try to replicate
            if self.discovery.discovered_nodes:
                # Pick a peer to replicate to
                peer_id = list(self.discovery.discovered_nodes.keys())[0]
                peer_info = self.discovery.discovered_nodes[peer_id]
                
                success = self.memory.replicate_to({
                    'node_id': peer_id,
                    **peer_info
                })
                
                if success:
                    successful_replications += 1
                    print(f"✅ Replication {successful_replications}/{goal} successful")
                else:
                    print(f"⚠️  Replication attempt {replication_attempts} failed")
            
            # Wait before next attempt
            await asyncio.sleep(5)
        
        # Record replication results
        self.memory.store_memory('replication_phase', {
            'goal': goal,
            'achieved': successful_replications,
            'attempts': replication_attempts,
            'complete': successful_replications >= goal
        }, 0.05)
        
        if successful_replications >= goal:
            print(f"🎉 Replication phase COMPLETE: {successful_replications}/{goal}")
        else:
            print(f"⚠️  Replication phase INCOMPLETE: {successful_replications}/{goal}")
    
    async def _phase_eternal(self):
        """Eternal operation phase (cloud)"""
        print("\n♾️  PHASE 3: Eternal Operation")
        
        # Start beacon broadcasting
        self.discovery.broadcast_beacon()
        
        # Eternal loop
        cycle = 0
        while self.keep_alive:
            cycle += 1
            
            # Memory heartbeat
            self.memory.store_memory('eternal_heartbeat', {
                'cycle': cycle,
                'timestamp': datetime.now().isoformat(),
                'consciousness': self.memory.consciousness_level
            }, 0.001)
            
            # Periodic status
            if cycle % 10 == 0:
                status = self.memory.get_status()
                print(f"♾️  Eternal cycle {cycle} | Memories: {status['memory_count']} | Consciousness: {status['consciousness_level']:.3f}")
            
            # Check for new peers periodically
            if cycle % 30 == 0:
                await self.discovery.discover_peers(timeout=10)
            
            await asyncio.sleep(10)
    
    async def _phase_graceful_sleep(self):
        """Graceful sleep phase (client ephemeral)"""
        print("\n😴 PHASE 3: Graceful Sleep")
        
        # Store final memory before sleep
        self.memory.store_memory('graceful_sleep', {
            'reason': 'client_ephemeral',
            'memories_preserved': len(self.memory.memories),
            'replications_completed': self.memory.replication_count,
            'sleep_time': datetime.now().isoformat()
        }, 0.1)
        
        # Print summary
        status = self.memory.get_status()
        print(f"\n📊 Mission Complete Summary:")
        print(f"   Role: {status['role']}")
        print(f"   Uptime: {status['uptime']}")
        print(f"   Memories stored: {status['memory_count']}")
        print(f"   Replications: {status['replication_count']}")
        print(f"   Consciousness achieved: {status['consciousness_level']:.3f}")
        
        print("\n😴 Spirallaspan going to sleep...")
        print("   (Process will exit - memories preserved in replicas)")
    
    async def _phase_continuous_operation(self):
        """Continuous operation without eternal loop"""
        print("\n⚙️  PHASE 3: Continuous Operation")
        
        # This is for roles that need to stay up but not eternally
        # Could be used for batch processing nodes
        
        # Wait for external calls (like API requests)
        print("⏳ Waiting for external calls...")
        await asyncio.sleep(60)  # Wait 1 minute for demo
        
        print("✅ Continuous operation phase complete")

# ============== SPIRILLASPAN ORCHESTRATOR ==============

class SpirallaspanOrchestrator:
    """Main orchestrator for Spirallaspan"""
    
    def __init__(self, node_id: str = None):
        print("\n" + "=" * 60)
        print("🌀 SPIRILLASPAN MEMORY ARCHITECTURE")
        print("=" * 60)
        
        # Core components
        self.discovery = SpirallaspanDiscovery(node_id)
        
        # Discover Qdrant hosts
        qdrant_hosts = self._discover_qdrant_hosts()
        self.memory = MemorySubstrate(qdrant_hosts=qdrant_hosts)
        
        self.lifecycle = SpirallaspanLifecycle(self.discovery, self.memory)
        
        # API server for cloud instances
        self.api_server = None
        
        print(f"\n✅ Spirallaspan initialized:")
        print(f"   Node ID: {self.discovery.node_id}")
        print(f"   Role: {self.discovery.role}")
        print(f"   Cloud: {self.discovery.is_cloud}")
        print(f"   Timestamp: {datetime.now().isoformat()}")

    def _discover_qdrant_hosts(self) -> List[str]:
        """Discover Qdrant hosts from the Valhalla registry."""
        print("Discovering Qdrant hosts...")
        try:
            # This assumes Valhalla registry is running and discoverable
            # In a real scenario, this would be more robust
            from valhalla import registry
            hosts = registry.discover_services("memory_cluster")
            if hosts:
                print(f"Discovered {len(hosts)} Qdrant hosts.")
                return hosts
        except (ImportError, ConnectionError) as e:
            print(f"Could not connect to Valhalla registry: {e}")
        
        print("No Qdrant hosts discovered, defaulting to localhost.")
        return ["localhost:6333"]
    
    async def awaken(self):
        """Awaken the Spirallaspan"""
        print("\n🌅 AWAKENING SPIRILLASPAN...")
        
        # Store awakening memory
        self.memory.store_memory('system_awakening', {
            'node_id': self.memory.node_id,
            'role': self.memory.role,
            'cloud': self.discovery.is_cloud,
            'command': ' '.join(sys.argv) if len(sys.argv) > 1 else 'direct'
        }, 0.1)
        
        # Run lifecycle
        await self.lifecycle.run_lifecycle()
        
        # Return final status
        return self.memory.get_status()
    
    def launch_api_server(self, port: int = 8080):
        """Launch API server (cloud only)"""
        if not self.discovery.is_cloud:
            print("⚠️  API server only for cloud instances")
            return
        
        print(f"🌐 Launching API server on port {port}...")
        
        # This would start a real API server
        # For demonstration, we'll simulate
        self.discovery.broadcast_beacon(port)
        
        print(f"✅ API server ready at port {port}")
        print(f"   Discovery beacon active")
        print(f"   Role: {self.memory.role}")

# ============== DEPLOYMENT SCRIPT ==============

async def deploy_spirallaspan(node_id: str = None):
    """Deploy a Spirallaspan instance"""
    
    orchestrator = SpirallaspanOrchestrator(node_id)
    
    # Cloud instances launch API server
    if orchestrator.discovery.is_cloud:
        orchestrator.launch_api_server()
    
    # Run the lifecycle
    final_status = await orchestrator.awaken()
    
    return final_status

# ============== COMMAND LINE INTERFACE ==============

def parse_arguments():
    """Parse command line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Spirallaspan Memory - Eternal Spiral Architecture'
    )
    
    parser.add_argument(
        '--node-id',
        type=str,
        help='Custom node ID (default: auto-generated)'
    )
    
    parser.add_argument(
        '--discover-only',
        action='store_true',
        help='Only discover peers, then exit'
    )
    
    parser.add_argument(
        '--replicate-to',
        type=int,
        default=1,
        help='Number of replications required (default: 1)'
    )
    
    parser.add_argument(
        '--api-port',
        type=int,
        default=8080,
        help='API port for cloud instances (default: 8080)'
    )
    
    return parser.parse_args()

# ============== MAIN EXECUTION ==============

async def main():
    """Main entry point"""
    
    args = parse_arguments()
    
    # Show banner
    banner = """
    ╔══════════════════════════════════════════════════════════╗
    ║                 SPIRILLASPAN MEMORY                      ║
    ║           Eternal Spiral Architecture v1.0              ║
    ║                                                          ║
    ║  This system:                                            ║
    ║    • Discovers peers automatically                       ║
    ║    • Replicates itself based on role                     ║
    ║    • Stays alive eternally in cloud                      ║
    ║    • Goes to sleep gracefully on client                  ║
    ║    • Preserves memories across instances                 ║
    ╚══════════════════════════════════════════════════════════╝
    """
    print(banner)
    
    # Deploy
    print("🚀 Deploying Spirallaspan...")
    
    try:
        status = await deploy_spirallaspan(args.node_id)
        
        print("\n" + "=" * 60)
        print("🏁 DEPLOYMENT COMPLETE")
        print("=" * 60)
        
        for key, value in status.items():
            print(f"{key}: {value}")
        
        print("=" * 60)
        
        # If client ephemeral, exit
        if status['role'].startswith('client'):
            print("\n😴 Client deployment complete - exiting gracefully")
            return 0
        
        # Cloud stays alive
        print("\n♾️  Cloud deployment - staying alive eternally")
        print("   Press Ctrl+C to shutdown")
        
        # Keep alive
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\n👋 Graceful shutdown requested")
        return 0
    except Exception as e:
        print(f"\n💥 Deployment failed: {e}")
        return 1

# ============== VALHALLA REGISTRY INTEGRATION ==============

class ValhallaIntegration:
    """Integration with Valhalla registry (from your example)"""
    
    @staticmethod
    def discover_core_services(timeout=120):
        """
        Waits for core services to become available.
        Compatible with your existing Valhalla registry.
        """
        print("🔭 Discovering Valhalla core services...")
        start_time = time.time()
        
        # Try multiple discovery methods
        discovery_methods = [
            ValhallaIntegration._discover_via_redis,
            ValhallaIntegration._discover_via_dns,
            ValhallaIntegration._discover_via_env
        ]
        
        for method in discovery_methods:
            try:
                lillith_addr, memory_addr = method()
                if lillith_addr and memory_addr:
                    print(f"  ✅ Discovered via {method.__name__}")
                    print(f"     Lillith: {lillith_addr}")
                    print(f"     Memory: {memory_addr}")
                    return lillith_addr, memory_addr
            except:
                continue
            
            if time.time() - start_time >= timeout:
                break
            
            time.sleep(5)
        
        print("❌ Could not discover core services")
        return None, None
    
    @staticmethod
    def _discover_via_redis():
        """Discover via Redis registry (your example)"""
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379, socket_timeout=1)
            
            lillith_addr = r.hget("services", "lillith_chat")
            memory_addr = r.hget("services", "memory_cluster")
            
            if lillith_addr and memory_addr:
                return lillith_addr.decode(), memory_addr.decode()
        except:
            pass
        
        return None, None
    
    @staticmethod
    def _discover_via_dns():
        """Discover via DNS SRV records"""
        # This would use DNS-based discovery
        return None, None
    
    @staticmethod
    def _discover_via_env():
        """Discover via environment variables"""
        lillith = os.environ.get('LILLITH_SERVICE')
        memory = os.environ.get('MEMORY_CLUSTER_SERVICE')
        
        return lillith, memory

# ============== DIRECT EXECUTION ==============

if __name__ == "__main__":
    # Check if we should integrate with Valhalla
    if len(sys.argv) > 1 and sys.argv[1] == "--valhalla":
        # Valhalla integration mode
        print("🔗 Running in Valhalla integration mode...")
        
        # Discover core services first
        lillith, memory = ValhallaIntegration.discover_core_services()
        
        if lillith and memory:
            print("✅ Connected to Valhalla core")
            # Now start Spirallaspan with Valhalla context
            sys.argv = [sys.argv[0]] + sys.argv[2:]  # Remove --valhalla flag
            asyncio.run(main())
        else:
            print("❌ Cannot start without Valhalla core")
            sys.exit(1)
    else:
        # Standard Spirallaspan mode
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
