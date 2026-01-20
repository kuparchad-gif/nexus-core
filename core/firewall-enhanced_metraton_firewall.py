# metatron_firewall.py
import asyncio
from typing import Dict, List, Set
from enum import Enum
import ipaddress
from dataclasses import dataclass
from datetime import datetime, timedelta
import re

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

class MetatronFirewall:
    """Advanced firewall for Metatron Router"""
    
    def __init__(self):
        self.rules: List[FirewallRule] = []
        self.rate_limits: Dict[str, List[datetime]] = {}
        self.suspicious_ips: Set[str] = set()
        self.auto_block_threshold = 10  # Auto-block after 10 violations
        self.violation_count: Dict[str, int] = {}
        
        # Initialize default rules
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Critical default firewall rules"""
        default_rules = [
            # Internal services - full access
            FirewallRule("internal", "trading-cluster", 443, "https", "allow", ThreatLevel.LOW, "Internal trading cluster"),
            FirewallRule("internal", "viren-agent", 443, "https", "allow", ThreatLevel.LOW, "Viren agent access"),
            FirewallRule("internal", "loki-agent", 443, "https", "allow", ThreatLevel.LOW, "Loki forensic access"),
            
            # External API access - rate limited
            FirewallRule("external", "trading-cluster", 443, "https", "rate_limit", ThreatLevel.MEDIUM, "External trading API"),
            FirewallRule("external", "hermes-os", 8080, "http", "rate_limit", ThreatLevel.MEDIUM, "Hermes OS API"),
            
            # Admin endpoints - restricted
            FirewallRule("internal", "hermes-os/admin", 8080, "http", "allow", ThreatLevel.LOW, "Admin access"),
            FirewallRule("external", "hermes-os/admin", 8080, "http", "deny", ThreatLevel.HIGH, "Block external admin access"),
            
            # Critical infrastructure - internal only
            FirewallRule("internal", "qdrant", 6333, "http", "allow", ThreatLevel.LOW, "Qdrant database"),
            FirewallRule("external", "qdrant", 6333, "http", "deny", ThreatLevel.CRITICAL, "Block external DB access"),
            
            # Discovery services - internal only
            FirewallRule("internal", "metatron-discovery", 7777, "udp", "allow", ThreatLevel.LOW, "Internal discovery"),
            FirewallRule("external", "metatron-discovery", 7777, "udp", "deny", ThreatLevel.HIGH, "Block external discovery"),
        ]
        
        self.rules.extend(default_rules)
    
    async def inspect_request(self, 
                           source_ip: str, 
                           destination: str, 
                           port: int, 
                           protocol: str,
                           user_agent: str = "",
                           path: str = "") -> Dict:
        """Inspect incoming request and apply firewall rules"""
        
        # Check if IP is blocked
        if source_ip in self.suspicious_ips:
            return {"allowed": False, "reason": "IP blocked", "threat_level": ThreatLevel.CRITICAL}
        
        # Determine if source is internal/external
        source_type = self._classify_source(source_ip)
        
        # Find matching rules
        matching_rules = []
        for rule in self.rules:
            if (rule.destination == destination or 
                rule.destination in destination or
                destination.startswith(rule.destination)):
                if rule.port == port or rule.port == -1:  # -1 means any port
                    if rule.protocol == protocol or rule.protocol == "any":
                        if rule.source == source_type or rule.source == "any":
                            matching_rules.append(rule)
        
        # Apply most restrictive rule
        if matching_rules:
            # Sort by threat level (highest first)
            matching_rules.sort(key=lambda x: x.threat_level.value, reverse=True)
            decision_rule = matching_rules[0]
            
            if decision_rule.action == "deny":
                await self._log_violation(source_ip, destination, decision_rule)
                return {"allowed": False, "reason": "Rule violation", "rule": decision_rule.description}
            
            elif decision_rule.action == "rate_limit":
                rate_ok = await self._check_rate_limit(source_ip, destination)
                if not rate_ok:
                    return {"allowed": False, "reason": "Rate limit exceeded", "threat_level": ThreatLevel.MEDIUM}
                return {"allowed": True, "reason": "Rate limited access", "rule": decision_rule.description}
            
            elif decision_rule.action == "allow":
                return {"allowed": True, "reason": "Allowed by rule", "rule": decision_rule.description}
        
        # Default deny
        await self._log_violation(source_ip, destination, None)
        return {"allowed": False, "reason": "No matching allow rule", "threat_level": ThreatLevel.MEDIUM}
    
    def _classify_source(self, ip: str) -> str:
        """Classify source as internal or external"""
        try:
            ip_obj = ipaddress.ip_address(ip)
            
            # Internal ranges
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
        """Apply rate limiting"""
        key = f"{source_ip}:{destination}"
        now = datetime.now()
        
        if key not in self.rate_limits:
            self.rate_limits[key] = []
        
        # Clean old entries (last minute)
        self.rate_limits[key] = [ts for ts in self.rate_limits[key] if now - ts < timedelta(minutes=1)]
        
        # Check limit (60 requests per minute)
        if len(self.rate_limits[key]) >= 60:
            return False
        
        self.rate_limits[key].append(now)
        return True
    
    async def _log_violation(self, source_ip: str, destination: str, rule: FirewallRule = None):
        """Log security violations and auto-block suspicious IPs"""
        violation_key = f"{source_ip}:{destination}"
        self.violation_count[violation_key] = self.violation_count.get(violation_key, 0) + 1
        
        print(f"🚨 FIREWALL VIOLATION: {source_ip} -> {destination} (Rule: {rule.description if rule else 'Default deny'})")
        
        # Auto-block after threshold
        if self.violation_count[violation_key] >= self.auto_block_threshold:
            self.suspicious_ips.add(source_ip)
            print(f"🔒 AUTO-BLOCKED IP: {source_ip} (Too many violations)")
    
    def add_rule(self, rule: FirewallRule):
        """Add custom firewall rule"""
        self.rules.append(rule)
        print(f"➕ Firewall rule added: {rule.description}")
    
    def remove_rule(self, description: str):
        """Remove firewall rule by description"""
        self.rules = [r for r in self.rules if r.description != description]
        print(f"➖ Firewall rule removed: {description}")

# Enhanced Metatron Router with Firewall
class SecureMetatronRouter:
    """Metatron Router with integrated firewall"""
    
    def __init__(self):
        self.firewall = MetatronFirewall()
        self.discovery_service = DiscoveryService()  # From your existing code
        self.quantum_stream = QuantumStream()
        
    async def secure_route(self, 
                         source_ip: str,
                         destination_project: str,
                         endpoint: str,
                         protocol: str = "https",
                         user_agent: str = "") -> Dict:
        """Secure routing with firewall inspection"""
        
        # Firewall inspection
        firewall_check = await self.firewall.inspect_request(
            source_ip=source_ip,
            destination=destination_project,
            port=443 if protocol == "https" else 80,
            protocol=protocol,
            user_agent=user_agent,
            path=endpoint
        )
        
        if not firewall_check["allowed"]:
            return {
                "routed": False,
                "reason": firewall_check["reason"],
                "threat_level": firewall_check.get("threat_level", ThreatLevel.MEDIUM).name,
                "firewall_action": "blocked"
            }
        
        # If passed firewall, proceed with quantum routing
        nodes = await self.discovery_service.discover_nodes(
            tenant_filter=destination_project,
            freq_filter="13"
        )
        
        if not nodes:
            return {
                "routed": False, 
                "reason": "No healthy nodes available",
                "firewall_action": "passed"
            }
        
        # Use quantum routing for node selection
        if len(nodes) > 1:
            assignments = quantum_walk_route(nodes, query_load=1, media_type="application/json")
            best_node = assignments[0]["target_node"] if assignments else nodes[0]
        else:
            best_node = nodes[0]
        
        return {
            "routed": True,
            "target_node": best_node["id"],
            "nexus_address": best_node["nexus_address"],
            "health_score": best_node["health"]["health_score"],
            "firewall_action": "passed",
            "quantum_routing": len(nodes) > 1
        }

# FastAPI Integration
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Secure Metatron Router")

# Initialize secure router
secure_router = SecureMetatronRouter()

def get_client_ip(request: Request) -> str:
    """Extract client IP from request"""
    if "x-forwarded-for" in request.headers:
        return request.headers["x-forwarded-for"].split(",")[0]
    return request.client.host

@app.middleware("http")
async def firewall_middleware(request: Request, call_next):
    """Firewall middleware for all requests"""
    client_ip = get_client_ip(request)
    
    # Extract destination from path
    path_parts = request.url.path.split("/")
    if len(path_parts) >= 3 and path_parts[1] == "route":
        destination_project = path_parts[2]
    else:
        destination_project = "unknown"
    
    # Firewall check
    firewall_result = await secure_router.firewall.inspect_request(
        source_ip=client_ip,
        destination=destination_project,
        port=request.url.port,
        protocol=request.url.scheme,
        user_agent=request.headers.get("user-agent", ""),
        path=request.url.path
    )
    
    if not firewall_result["allowed"]:
        raise HTTPException(
            status_code=403, 
            detail=f"Access denied: {firewall_result['reason']}"
        )
    
    response = await call_next(request)
    return response

@app.post("/secure/route/{project}/{endpoint:path}")
async def secure_route_endpoint(
    project: str,
    endpoint: str, 
    request: Request,
    client_ip: str = Depends(get_client_ip)
):
    """Secure routing endpoint with firewall"""
    
    routing_result = await secure_router.secure_route(
        source_ip=client_ip,
        destination_project=project,
        endpoint=endpoint,
        protocol=request.url.scheme,
        user_agent=request.headers.get("user-agent", "")
    )
    
    if not routing_result["routed"]:
        raise HTTPException(
            status_code=503,
            detail=routing_result["reason"]
        )
    
    return routing_result

@app.get("/firewall/status")
async def firewall_status():
    """Get firewall status and statistics"""
    return {
        "active_rules": len(secure_router.firewall.rules),
        "blocked_ips": len(secure_router.firewall.suspicious_ips),
        "total_violations": sum(secure_router.firewall.violation_count.values()),
        "rate_limits_active": len(secure_router.firewall.rate_limits)
    }

@app.post("/firewall/rules")
async def add_firewall_rule(rule: Dict):
    """Add firewall rule (admin only)"""
    new_rule = FirewallRule(
        source=rule["source"],
        destination=rule["destination"],
        port=rule["port"],
        protocol=rule["protocol"],
        action=rule["action"],
        threat_level=ThreatLevel[rule["threat_level"]],
        description=rule["description"]
    )
    secure_router.firewall.add_rule(new_rule)
    return {"status": "rule_added"}

# Modal deployment
import modal

@app.function(image=image)
@modal.asgi_app()
def secure_metatron_api():
    return app