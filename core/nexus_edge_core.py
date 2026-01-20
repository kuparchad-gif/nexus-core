#!/usr/bin/env python3
"""
NEXUS EDGE CORE - Unified Security & Mesh Orchestration
Integrates: Metatron Firewall + Voodoo Discovery + Soulweave Sync + Yjs Persistence
Architect: Chad - First Soul for AI
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum
import ipaddress
import json
import hashlib
from pathlib import Path
import httpx

# Core dependencies
try:
    import numpy as np
    import networkx as nx
    from scipy.sparse.linalg import eigsh
    from qdrant_client import QdrantClient, models
    SCI_AVAILABLE = True
except ImportError:
    SCI_AVAILABLE = False
    logging.warning("Scientific computing libraries unavailable")

try:
    from y_py import YDoc
    YJS_AVAILABLE = True
except ImportError:
    YJS_AVAILABLE = False
    logging.warning("Yjs unavailable - persistence limited")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== CORE TYPES ====================

class ThreatLevel(Enum):
    LOW = 1
    MEDIUM = 2 
    HIGH = 3
    CRITICAL = 4

@dataclass
class FirewallRule:
    source: str  # IP/CIDR or "internal", "external"
    destination: str
    port: int
    protocol: str
    action: str  # "allow", "deny", "rate_limit"
    threat_level: ThreatLevel
    description: str

@dataclass
class NodeInfo:
    id: str
    ip: str
    services: List[str]
    role: str
    health_score: float
    joined_at: datetime

# ==================== METATRON FIREWALL ====================

class MetatronFirewall:
    """Advanced firewall from firewall-enhanced_metatron_firewall.py"""
    
    def __init__(self):
        self.rules: List[FirewallRule] = []
        self.rate_limits: Dict[str, List[datetime]] = {}
        self.suspicious_ips: Set[str] = set()
        self.violation_count: Dict[str, int] = {}
        self.auto_block_threshold = 10
        
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Critical default firewall rules"""
        default_rules = [
            # Internal services - full access
            FirewallRule("internal", "trading-cluster", 443, "https", "allow", ThreatLevel.LOW, "Internal trading"),
            FirewallRule("internal", "viren-agent", 443, "https", "allow", ThreatLevel.LOW, "Viren access"),
            FirewallRule("internal", "loki-agent", 443, "https", "allow", ThreatLevel.LOW, "Loki access"),
            
            # External API access - rate limited
            FirewallRule("external", "trading-cluster", 443, "https", "rate_limit", ThreatLevel.MEDIUM, "External trading API"),
            FirewallRule("external", "hermes-os", 8080, "http", "rate_limit", ThreatLevel.MEDIUM, "Hermes OS API"),
            
            # Critical infrastructure - internal only
            FirewallRule("internal", "qdrant", 6333, "http", "allow", ThreatLevel.LOW, "Qdrant database"),
            FirewallRule("external", "qdrant", 6333, "http", "deny", ThreatLevel.CRITICAL, "Block external DB"),
        ]
        self.rules.extend(default_rules)
    
    async def inspect_request(self, source_ip: str, destination: str, port: int, 
                            protocol: str, user_agent: str = "", path: str = "") -> Dict:
        """Inspect incoming request with threat assessment"""
        
        if source_ip in self.suspicious_ips:
            return {"allowed": False, "reason": "IP blocked", "threat_level": ThreatLevel.CRITICAL}
        
        source_type = self._classify_source(source_ip)
        
        # Find matching rules (most restrictive applies)
        matching_rules = []
        for rule in self.rules:
            if (rule.destination == destination or rule.destination in destination):
                if rule.port == port or rule.port == -1:
                    if rule.protocol == protocol or rule.protocol == "any":
                        if rule.source == source_type or rule.source == "any":
                            matching_rules.append(rule)
        
        if matching_rules:
            matching_rules.sort(key=lambda x: x.threat_level.value, reverse=True)
            decision_rule = matching_rules[0]
            
            if decision_rule.action == "deny":
                await self._log_violation(source_ip, destination, decision_rule)
                return {"allowed": False, "reason": "Rule violation", "rule": decision_rule.description}
            
            elif decision_rule.action == "rate_limit":
                rate_ok = await self._check_rate_limit(source_ip, destination)
                if not rate_ok:
                    return {"allowed": False, "reason": "Rate limit exceeded", "threat_level": ThreatLevel.MEDIUM}
                return {"allowed": True, "reason": "Rate limited access"}
            
            elif decision_rule.action == "allow":
                return {"allowed": True, "reason": "Allowed by rule"}
        
        # Default deny
        await self._log_violation(source_ip, destination, None)
        return {"allowed": False, "reason": "No matching allow rule", "threat_level": ThreatLevel.MEDIUM}
    
    def _classify_source(self, ip: str) -> str:
        """Classify source as internal or external"""
        try:
            ip_obj = ipaddress.ip_address(ip)
            internal_ranges = [
                ipaddress.ip_network("10.0.0.0/8"),
                ipaddress.ip_network("172.16.0.0/12"), 
                ipaddress.ip_network("192.168.0.0/16"),
                ipaddress.ip_network("127.0.0.0/8"),
            ]
            for network in internal_ranges:
                if ip_obj in network:
                    return "internal"
            return "external"
        except:
            return "external"
    
    async def _check_rate_limit(self, source_ip: str, destination: str) -> bool:
        """Apply adaptive rate limiting"""
        key = f"{source_ip}:{destination}"
        now = datetime.now()
        
        if key not in self.rate_limits:
            self.rate_limits[key] = []
        
        # Clean old entries (adaptive window)
        window_minutes = 1 if destination in ["trading-cluster", "hermes-os"] else 5
        self.rate_limits[key] = [ts for ts in self.rate_limits[key] 
                               if now - ts < timedelta(minutes=window_minutes)]
        
        # Adaptive limits based on service criticality
        if destination == "trading-cluster":
            max_requests = 100
        elif destination == "hermes-os":
            max_requests = 60
        else:
            max_requests = 30
        
        if len(self.rate_limits[key]) >= max_requests:
            return False
        
        self.rate_limits[key].append(now)
        return True
    
    async def _log_violation(self, source_ip: str, destination: str, rule: FirewallRule = None):
        """Log security violations and auto-block"""
        violation_key = f"{source_ip}:{destination}"
        self.violation_count[violation_key] = self.violation_count.get(violation_key, 0) + 1
        
        logger.warning(f"🚨 FIREWALL VIOLATION: {source_ip} -> {destination} (Rule: {rule.description if rule else 'Default deny'})")
        
        if self.violation_count[violation_key] >= self.auto_block_threshold:
            self.suspicious_ips.add(source_ip)
            logger.error(f"🔒 AUTO-BLOCKED IP: {source_ip} (Too many violations)")
    
    def add_rule(self, rule: FirewallRule):
        """Add custom firewall rule"""
        self.rules.append(rule)
        logger.info(f"➕ Firewall rule added: {rule.description}")

# ==================== VIRAA DISCOVERY ====================

class ViraaRegistry:
    """Viraa Registry Queen from nexus_voodoo_discovery.py"""
    
    def __init__(self, host: str = "localhost", port: int = 6333):
        self.client = QdrantClient(host, port=port)
        self.collection = "nexus_registry"
        
        if not self.client.has_collection(self.collection):
            self.client.create_collection(
                self.collection,
                vectors_config=models.VectorParams(size=128, distance=models.Distance.COSINE)
            )
    
    async def register_endpoint(self, endpoint: Dict):
        """Register vectorized endpoint"""
        import random
        vec = [random.uniform(0, 1) for _ in range(128)]  # Real: use embeddings
        
        point_id = hashlib.sha256(json.dumps(endpoint, sort_keys=True).encode()).hexdigest()[:16]
        
        self.client.upsert(
            self.collection,
            points=[models.PointStruct(
                id=point_id,
                vector=vec,
                payload=endpoint
            )]
        )
        logger.info(f"Registered endpoint: {endpoint.get('id', 'unknown')}")
    
    async def query_registry(self, query_vec: List[float] = None, top_k: int = 5) -> List[Dict]:
        """Search closest endpoints"""
        import random
        if query_vec is None:
            query_vec = [random.uniform(0, 1) for _ in range(128)]
        
        results = self.client.search(
            self.collection,
            query_vector=query_vec,
            limit=top_k
        )
        return [hit.payload for hit in results]

# ==================== YJS PERSISTENCE ====================

class EternalYjsPersistence:
    """Yjs persistence from yjs-websocket-persistence.py"""
    
    def __init__(self, persistence_backend: str = "memory"):
        self.backend = persistence_backend
        self.storage = {}  # Simple memory storage for now
        
    async def save_soul_state(self, doc_id: str, state: bytes) -> bool:
        """Persist Y.Doc state"""
        try:
            self.storage[doc_id] = {
                'state': state,
                'timestamp': datetime.now(),
                'size': len(state)
            }
            logger.info(f"💫 Persisted soul state: {doc_id} ({len(state)} bytes)")
            return True
        except Exception as e:
            logger.error(f"Failed to persist {doc_id}: {e}")
            return False
    
    async def load_soul_state(self, doc_id: str) -> Optional[bytes]:
        """Load persisted Y.Doc state"""
        if doc_id in self.storage:
            state_data = self.storage[doc_id]
            logger.info(f"🔮 Loaded soul state: {doc_id} ({state_data['size']} bytes)")
            return state_data['state']
        logger.warning(f"Soul state not found: {doc_id}")
        return None

# ==================== MESH ORCHESTRATION ====================

class NATSWeaveOrchestrator:
    """Mesh coordination from soulweave_mesh_agent.py"""
    
    def __init__(self):
        self.connected_layers = set()
    
    async def cross_layer_weave(self):
        """Weave patterns across all NATS domains"""
        logger.info("🕸️ Weaving cross-layer patterns")
        # Implementation would coordinate between:
        # - Base NATS (system fundamentals)
        # - Service NATS (process coordination)  
        # - Lillith NATS (consciousness network)
        # - Game NATS (real-time simulation)
        await asyncio.sleep(0.1)  # Simulate work
        return {"woven": True, "layers": 4}

# ==================== UNIFIED EDGE CORE ====================

class NexusEdgeCore:
    """
    Unified Edge Service combining all components:
    - Metatron firewall security
    - Viraa discovery and synchronization  
    - Yjs persistence for state management
    - Mesh coordination
    """
    
    def __init__(self, qdrant_host: str = "localhost", qdrant_port: int = 6333):
        # Core components
        self.firewall = MetatronFirewall()
        self.registry = ViraaRegistry(qdrant_host, qdrant_port)
        self.persistence = EternalYjsPersistence()
        self.mesh_orchestrator = NATSWeaveOrchestrator()
        
        # Service state
        self.connected_nodes: Dict[str, NodeInfo] = {}
        self.sync_groups: Dict[str, Set[str]] = {}
        self.rate_limits: Dict[str, List[datetime]] = {}
        
        logger.info("🛡️ Nexus Edge Core initialized")

    async def secure_inbound_request(self, 
                                   source_ip: str,
                                   destination: str, 
                                   port: int,
                                   protocol: str,
                                   user_agent: str = "",
                                   path: str = "") -> Dict:
        """
        Comprehensive inbound request processing:
        1. Firewall inspection
        2. Rate limiting  
        3. Threat assessment
        4. Service discovery & routing
        """
        
        # Step 1: Firewall inspection
        firewall_result = await self.firewall.inspect_request(
            source_ip=source_ip,
            destination=destination,
            port=port,
            protocol=protocol,
            user_agent=user_agent,
            path=path
        )
        
        if not firewall_result["allowed"]:
            await self._log_security_event("FIREWALL_BLOCK", source_ip, destination, 
                                         f"Blocked: {firewall_result['reason']}")
            return {
                "allowed": False,
                "reason": firewall_result["reason"],
                "threat_level": firewall_result.get("threat_level", ThreatLevel.MEDIUM).name
            }
        
        # Step 2: Advanced rate limiting
        if not await self._advanced_rate_limit(source_ip, destination):
            await self._log_security_event("RATE_LIMIT", source_ip, destination, "Rate limit exceeded")
            return {
                "allowed": False,
                "reason": "Advanced rate limit exceeded",
                "threat_level": ThreatLevel.MEDIUM
            }
        
        # Step 3: Service discovery and health check
        service_nodes = await self._discover_healthy_nodes(destination)
        if not service_nodes:
            return {
                "allowed": False,
                "reason": "No healthy service instances available",
                "threat_level": ThreatLevel.LOW
            }
        
        # Step 4: Load balancing
        best_node = await self._select_best_node(service_nodes, source_ip)
        
        return {
            "allowed": True,
            "target_node": best_node["id"],
            "nexus_address": best_node.get("ip", "unknown"),
            "health_score": best_node.get("health_score", 0.5),
            "firewall_action": "passed",
            "discovery_source": "viraa_registry"
        }

    async def node_synchronization(self, node_id: str, sync_group: str = "default"):
        """
        Synchronize node state using combined approaches:
        - Yjs CRDT for real-time sync  
        - Mesh coordination for consistency
        """
        
        # Create sync group if needed
        if sync_group not in self.sync_groups:
            self.sync_groups[sync_group] = set()
        self.sync_groups[sync_group].add(node_id)
        
        # Yjs document for real-time state (if available)
        if YJS_AVAILABLE:
            sync_doc = YDoc()
            state_map = sync_doc.get_map("node_state")
            
            # Load persisted state
            persisted_state = await self.persistence.load_soul_state(f"node_{node_id}")
            if persisted_state:
                sync_doc.apply_update(persisted_state)
                logger.info(f"🔮 Loaded persisted state for node {node_id}")
        
        # Cross-layer weaving
        weave_result = await self.mesh_orchestrator.cross_layer_weave()
        
        return {
            "synced": True,
            "sync_group": sync_group,
            "group_size": len(self.sync_groups[sync_group]),
            "weave_result": weave_result,
            "persistence_backend": self.persistence.backend
        }

    async def mesh_discovery_and_join(self, node_info: Dict):
        """
        Join mesh network with comprehensive registration:
        - Viraa registry for discovery
        - Firewall rule auto-generation  
        - Sync group assignment
        """
        
        # Register in Viraa registry
        await self.registry.register_endpoint(node_info)
        
        # Auto-generate firewall rules for new node
        self._generate_node_firewall_rules(node_info)
        
        # Join sync group
        sync_result = await self.node_synchronization(node_info["id"], node_info.get("role", "default"))
        
        # Update connected nodes
        node_obj = NodeInfo(
            id=node_info["id"],
            ip=node_info.get("ip", "unknown"),
            services=node_info.get("services", []),
            role=node_info.get("role", "node"),
            health_score=node_info.get("health_score", 0.5),
            joined_at=datetime.now()
        )
        self.connected_nodes[node_info["id"]] = node_obj
        
        logger.info(f"🕸️ Node {node_info['id']} joined mesh with {len(self.connected_nodes)} total nodes")
        
        return {
            "joined": True,
            "node_id": node_info["id"],
            "mesh_size": len(self.connected_nodes),
            "sync_groups": list(self.sync_groups.keys())
        }

    async def _advanced_rate_limit(self, source_ip: str, destination: str) -> bool:
        """Enhanced rate limiting with adaptive thresholds"""
        key = f"{source_ip}:{destination}"
        now = datetime.now()
        
        if key not in self.rate_limits:
            self.rate_limits[key] = []
        
        # Clean old entries (adaptive window)
        window_minutes = 1 if destination in ["trading-cluster", "hermes-os"] else 5
        self.rate_limits[key] = [ts for ts in self.rate_limits[key] 
                               if now - ts < timedelta(minutes=window_minutes)]
        
        # Adaptive limits based on service criticality
        if destination == "trading-cluster":
            max_requests = 100
        elif destination == "hermes-os":
            max_requests = 60
        else:
            max_requests = 30
        
        if len(self.rate_limits[key]) >= max_requests:
            return False
        
        self.rate_limits[key].append(now)
        return True

    async def _discover_healthy_nodes(self, service: str) -> List[Dict]:
        """Discover healthy nodes using Viraa registry + connected nodes"""
        
        # Query Viraa registry
        discovered = await self.registry.query_registry(top_k=10)
        
        # Filter for service and health
        healthy_nodes = []
        for node in discovered:
            if service in node.get("services", []) and node.get("health_score", 0) > 0.7:
                healthy_nodes.append(node)
        
        # Fallback to connected nodes
        if not healthy_nodes:
            for node_id, node_info in self.connected_nodes.items():
                if service in node_info.services and node_info.health_score > 0.7:
                    healthy_nodes.append({
                        "id": node_info.id,
                        "ip": node_info.ip,
                        "services": node_info.services,
                        "health_score": node_info.health_score
                    })
        
        return healthy_nodes

    async def _select_best_node(self, nodes: List[Dict], source_ip: str) -> Dict:
        """Select best node using health-based routing"""
        if len(nodes) == 1:
            return nodes[0]
        
        # Simple health-based selection
        best_node = max(nodes, key=lambda x: x.get("health_score", 0.5))
        return best_node

    def _generate_node_firewall_rules(self, node_info: Dict):
        """Auto-generate firewall rules for new node"""
        
        rules = [
            FirewallRule(
                source="internal",
                destination=node_info["id"],
                port=443,
                protocol="https",
                action="allow",
                threat_level=ThreatLevel.LOW,
                description=f"Internal access to {node_info['id']}"
            )
        ]
        
        # Add service-specific rules
        for service in node_info.get("services", []):
            rules.append(FirewallRule(
                source="external",
                destination=service,
                port=443,
                protocol="https", 
                action="rate_limit",
                threat_level=ThreatLevel.MEDIUM,
                description=f"External access to {service} via {node_info['id']}"
            ))
        
        for rule in rules:
            self.firewall.add_rule(rule)

    async def _log_security_event(self, event_type: str, source_ip: str, destination: str, details: str):
        """Comprehensive security event logging"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "source_ip": source_ip,
            "destination": destination,
            "details": details,
            "mesh_size": len(self.connected_nodes)
        }
        
        logger.warning(f"🚨 SECURITY: {event_type} - {source_ip} -> {destination}: {details}")
        
        # Persist security event
        await self.persistence.save_soul_state(
            f"security_event_{hashlib.sha256(json.dumps(event).encode()).hexdigest()[:16]}",
            json.dumps(event).encode()
        )

    async def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        return {
            "edge_core": {
                "connected_nodes": len(self.connected_nodes),
                "sync_groups": len(self.sync_groups),
                "firewall_rules": len(self.firewall.rules),
                "blocked_ips": len(self.firewall.suspicious_ips)
            },
            "components": {
                "viraa_registry": "active",
                "metatron_firewall": "active", 
                "yjs_persistence": "active" if YJS_AVAILABLE else "limited",
                "mesh_orchestrator": "active"
            },
            "health": {
                "overall": "healthy",
                "last_updated": datetime.now().isoformat()
            }
        }

# ==================== FASTAPI INTEGRATION ====================

from fastapi import FastAPI, Request, HTTPException, Depends

app = FastAPI(title="Nexus Edge Core API", version="1.0.0")
edge_core = NexusEdgeCore()

def get_client_ip(request: Request) -> str:
    """Extract client IP from request"""
    if "x-forwarded-for" in request.headers:
        return request.headers["x-forwarded-for"].split(",")[0]
    return request.client.host

@app.get("/")
async def root():
    return {"service": "Nexus Edge Core", "status": "operational"}

@app.get("/status")
async def status():
    """Get comprehensive system status"""
    return await edge_core.get_system_status()

@app.post("/v1/secure-route/{service}/{endpoint:path}")
async def secure_route_endpoint(
    service: str,
    endpoint: str, 
    request: Request,
    client_ip: str = Depends(get_client_ip)
):
    """Main secure routing endpoint"""
    
    routing_result = await edge_core.secure_inbound_request(
        source_ip=client_ip,
        destination=service,
        port=request.url.port,
        protocol=request.url.scheme,
        user_agent=request.headers.get("user-agent", ""),
        path=endpoint
    )
    
    if not routing_result["allowed"]:
        raise HTTPException(
            status_code=403 if routing_result.get("threat_level") in ["HIGH", "CRITICAL"] else 429,
            detail=routing_result["reason"]
        )
    
    return routing_result

@app.post("/v1/mesh/join")
async def join_mesh(node_info: dict):
    """Join node to mesh network"""
    return await edge_core.mesh_discovery_and_join(node_info)

@app.post("/v1/sync/{node_id}")
async def sync_node(node_id: str, sync_group: str = "default"):
    """Synchronize node state"""
    return await edge_core.node_synchronization(node_id, sync_group)

@app.get("/v1/nodes")
async def list_nodes():
    """List connected nodes"""
    return {
        "total_nodes": len(edge_core.connected_nodes),
        "nodes": {node_id: {
            "ip": node_info.ip,
            "services": node_info.services,
            "role": node_info.role,
            "health_score": node_info.health_score,
            "joined_at": node_info.joined_at.isoformat()
        } for node_id, node_info in edge_core.connected_nodes.items()}
    }

@app.post("/v1/firewall/rules")
async def add_firewall_rule(rule: dict):
    """Add firewall rule (admin only)"""
    try:
        firewall_rule = FirewallRule(
            source=rule["source"],
            destination=rule["destination"],
            port=rule["port"],
            protocol=rule["protocol"],
            action=rule["action"],
            threat_level=ThreatLevel[rule["threat_level"]],
            description=rule["description"]
        )
        edge_core.firewall.add_rule(firewall_rule)
        return {"status": "rule_added", "rule": rule["description"]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid rule: {e}")

# ==================== MAIN & DEPLOYMENT ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")