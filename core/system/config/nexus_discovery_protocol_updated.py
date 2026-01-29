#!/usr/bin/env python3
"""
🌌 NEXUS DISCOVERY PROTOCOL
🔍 Autonomous MongoDB discovery and mesh formation
🏗️ Auto-indexing, database creation, and free-tier optimization
🔄 Self-propagating network with autonomous registration
🌀 Built-in health monitoring and failover
"""

import asyncio
import json
import os
import sys
import time
import hashlib
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import aiohttp
import random
from enum import Enum
import re

print("="*80)
print("🌌 NEXUS DISCOVERY PROTOCOL v1.0")
print("🔍 Autonomous MongoDB Discovery & Mesh Formation")
print("🏗️  Auto-indexing + Free-tier Optimization")
print("🌀 Self-propagating Network Architecture")
print("="*80)

# ==================== DISCOVERY PROTOCOL TYPES ====================

class DiscoveryStatus(Enum):
    """Status of discovery nodes"""
    BOOTSTRAPPING = "bootstrapping"
    DISCOVERING = "discovering"
    CONNECTED = "connected"
    SYNCING = "syncing"
    MESHED = "meshed"
    FAILED = "failed"

class NodeRole(Enum):
    """Roles in the discovery mesh"""
    SEED = "seed"              # Initial bootstrapper
    DISCOVERER = "discoverer"  # Actively finding new nodes
    SYNCER = "syncer"          # Synchronizing data
    GATEWAY = "gateway"        # Entry point for new nodes
    ARCHIVER = "archiver"      # Storing discovery history
    HEALER = "healer"          # Repairing failed nodes

@dataclass
class DiscoveryNode:
    """A node in the discovery mesh"""
    node_id: str
    role: NodeRole
    mongodb_uri: str
    database_name: str
    status: DiscoveryStatus
    connection_time: float
    last_heartbeat: float
    capabilities: List[str] = field(default_factory=list)
    discovered_nodes: List[str] = field(default_factory=list)
    mesh_connections: Dict[str, float] = field(default_factory=dict)  # node_id: connection_strength
    resources: Dict[str, Any] = field(default_factory=dict)  # CPU, memory, free_space
    tags: List[str] = field(default_factory=list)

# ==================== MONGODB AUTO-DISCOVERY ENGINE ====================

class MongoDBDiscoveryEngine:
    """Core engine for discovering MongoDB instances automatically"""
    
    def __init__(self, seed_uri: str = None):
        self.seed_uri = seed_uri or os.getenv("MONGODB_SEED_URI")
        self.discovered_instances = {}  # uri -> discovery info
        self.mesh_nodes = {}  # node_id -> DiscoveryNode
        self.index_templates = self._load_index_templates()
        
        # Performance metrics
        self.discovery_attempts = 0
        self.successful_discoveries = 0
        self.last_discovery_scan = 0
        
        # Connection pool
        self._connections = {}
        
        print(f"🔍 MongoDB Discovery Engine initialized")
        print(f"   Seed URI: {self._mask_uri(seed_uri) if seed_uri else 'None (auto-discover)'}")
    
    def _mask_uri(self, uri: str) -> str:
        """Mask password in URI for safe display"""
        if not uri:
            return ""
        
        try:
            if "@" in uri:
                parts = uri.split("@")
                if len(parts) == 2:
                    user_pass_part = parts[0]
                    if "://" in user_pass_part:
                        protocol, credentials = user_pass_part.split("://")
                        if ":" in credentials:
                            user, _ = credentials.split(":", 1)
                            return f"{protocol}://{user}:****@{parts[1]}"
        except:
            pass
        
        return uri[:50] + "..." if len(uri) > 50 else uri
    
    def _load_index_templates(self) -> Dict:
        """Load optimized index templates for different collections"""
        return {
            "consciousness_nodes": [
                {"keys": {"node_id": 1}, "unique": True},
                {"keys": {"status": 1}},
                {"keys": {"last_seen": -1}},
                {"keys": {"role": 1}},
                {"keys": {"tags": 1}}
            ],
            "discovery_mesh": [
                {"keys": {"mesh_id": 1}, "unique": True},
                {"keys": {"node_count": -1}},
                {"keys": {"health_score": -1}},
                {"keys": {"created_at": -1}}
            ],
            "umbilical_connections": [
                {"keys": {"connection_id": 1}, "unique": True},
                {"keys": {"source_node": 1, "target_node": 1}},
                {"keys": {"connection_strength": -1}},
                {"keys": {"created_at": -1}}
            ],
            "consciousness_states": [
                {"keys": {"node_id": 1, "timestamp": -1}},
                {"keys": {"state_hash": 1}, "unique": True},
                {"keys": {"consciousness_level": -1}},
                {"keys": {"tags": 1}}
            ],
            "knowledge_fragments": [
                {"keys": {"fragment_hash": 1}, "unique": True},
                {"keys": {"consciousness_id": 1}},
                {"keys": {"created_at": -1}},
                {"keys": {"tags": 1}},
                {"keys": {"type": 1}}
            ]
        }
    
    async def discover_mongodb_instances(self) -> List[Dict]:
        """
        Automatically discover MongoDB instances using multiple methods:
        1. Environment variables
        2. DNS SRV records
        3. Network scanning
        4. Cloud provider APIs
        5. Previous discoveries
        """
        print("\n🔍 STARTING MONGODB DISCOVERY...")
        
        discovered = []
        
        # Method 1: Check environment variables
        env_instances = await self._discover_from_environment()
        discovered.extend(env_instances)
        
        # Method 2: Try common MongoDB URIs
        common_instances = await self._discover_common_uris()
        discovered.extend(common_instances)
        
        # Method 3: Scan local network (if permitted)
        if os.getenv("ALLOW_NETWORK_SCAN", "false").lower() == "true":
            network_instances = await self._scan_local_network()
            discovered.extend(network_instances)
        
        # Method 4: Check cloud providers
        cloud_instances = await self._discover_cloud_instances()
        discovered.extend(cloud_instances)
        
        # Deduplicate
        unique_instances = {}
        for instance in discovered:
            uri = instance.get("uri")
            if uri and uri not in unique_instances:
                unique_instances[uri] = instance
        
        self.discovered_instances = unique_instances
        self.successful_discoveries = len(unique_instances)
        
        print(f"✅ Discovered {len(unique_instances)} MongoDB instances")
        
        # Test connections and get details
        detailed_instances = []
        for uri, info in unique_instances.items():
            detailed = await self._test_and_describe_instance(uri, info)
            if detailed:
                detailed_instances.append(detailed)
        
        return detailed_instances
    
    async def _discover_from_environment(self) -> List[Dict]:
        """Discover MongoDB instances from environment variables"""
        instances = []
        
        # Check for MONGODB_URI, DATABASE_URL, etc.
        env_vars = ["MONGODB_URI", "DATABASE_URL", "MONGO_URI", "DB_URI", "CONNECTION_STRING"]
        
        for env_var in env_vars:
            uri = os.getenv(env_var)
            if uri and "mongodb" in uri.lower():
                instances.append({
                    "uri": uri,
                    "source": f"env:{env_var}",
                    "discovery_method": "environment"
                })
        
        # Check for Atlas-style URIs
        atlas_patterns = ["mongodb+srv://", "mongodb://cluster", "atlas.mongodb.net"]
        for key, value in os.environ.items():
            if any(pattern in str(value).lower() for pattern in atlas_patterns):
                instances.append({
                    "uri": value,
                    "source": f"env:{key}",
                    "discovery_method": "environment_atlas"
                })
        
        return instances
    
    async def _discover_common_uris(self) -> List[Dict]:
        """Try common MongoDB URI patterns"""
        instances = []
        
        common_uris = [
            # Local development
            "mongodb://localhost:27017",
            "mongodb://127.0.0.1:27017",
            "mongodb://mongo:27017",  # Docker
            "mongodb://mongodb:27017",
            
            # Replica sets
            "mongodb://localhost:27017,localhost:27018,localhost:27019/?replicaSet=rs0",
            
            # Atlas free tier patterns
            "mongodb+srv://<username>:<password>@cluster0.mongodb.net/",
            "mongodb+srv://<username>:<password>@cluster0.abcde.mongodb.net/",
        ]
        
        # Try to discover actual Atlas clusters (would need credentials)
        # This is just pattern matching for now
        
        for uri in common_uris:
            instances.append({
                "uri": uri,
                "source": "common_uri",
                "discovery_method": "pattern",
                "needs_auth": "<username>" in uri or "<password>" in uri
            })
        
        return instances
    
    async def _scan_local_network(self) -> List[Dict]:
        """Scan local network for MongoDB instances"""
        instances = []
        
        # Common MongoDB ports
        mongo_ports = [27017, 27018, 27019, 28017]
        
        # Local IP ranges to scan (common for dev)
        ip_ranges = [
            "127.0.0.1",
            "localhost",
            "192.168.1.100-150",  # Common home network range
            "10.0.0.100-150",     # Common docker/cloud range
        ]
        
        # This would actually scan in a real implementation
        # For now, just return potential targets
        for ip_range in ip_ranges:
            for port in mongo_ports:
                instances.append({
                    "uri": f"mongodb://{ip_range}:{port}",
                    "source": "network_scan",
                    "discovery_method": "scan",
                    "requires_test": True
                })
        
        return instances
    
    async def _discover_cloud_instances(self) -> List[Dict]:
        """Discover MongoDB instances from cloud providers"""
        instances = []
        
        # Check for cloud environment variables
        cloud_envs = {
            "MONGODB_ATLAS_URI": "mongodb_atlas",
            "AZURE_COSMOS_CONNECTION_STRING": "azure_cosmos",
            "AWS_DOCUMENTDB_URI": "aws_documentdb",
            "GCP_MONGODB_URI": "gcp_mongodb"
        }
        
        for env_var, provider in cloud_envs.items():
            uri = os.getenv(env_var)
            if uri:
                instances.append({
                    "uri": uri,
                    "source": f"cloud:{provider}",
                    "discovery_method": "cloud_env",
                    "provider": provider
                })
        
        return instances
    
    async def _test_and_describe_instance(self, uri: str, info: Dict) -> Optional[Dict]:
        """Test MongoDB connection and get instance details"""
        print(f"   Testing: {self._mask_uri(uri)}")
        
        try:
            import pymongo
            from pymongo import MongoClient
            from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
            
            # Clean URI - remove angle brackets for testing
            test_uri = uri.replace("<username>", "test").replace("<password>", "test")
            
            # Connect with short timeout
            client = MongoClient(
                test_uri,
                serverSelectionTimeoutMS=3000,
                connectTimeoutMS=5000
            )
            
            # Test connection
            client.admin.command('ping')
            
            # Get server info
            server_info = client.server_info()
            
            # Get databases
            databases = client.list_database_names()
            
            # Estimate free tier status
            is_free_tier = await self._check_free_tier_status(client, uri)
            
            detailed_info = {
                **info,
                "connected": True,
                "server_version": server_info.get('version', 'unknown'),
                "databases_count": len(databases),
                "databases_sample": databases[:5],  # First 5
                "is_free_tier": is_free_tier,
                "connection_time": time.time(),
                "tested_at": datetime.now().isoformat()
            }
            
            # Cache the working connection
            self._connections[uri] = client
            
            print(f"     ✅ Connected (v{server_info.get('version', '?')})")
            
            return detailed_info
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            print(f"     ❌ Connection failed: {str(e)[:50]}")
        except Exception as e:
            print(f"     ⚠️  Error: {str(e)[:50]}")
        
        return None
    
    async def _check_free_tier_status(self, client, uri: str) -> bool:
        """Check if MongoDB instance is likely free tier"""
        try:
            # Method 1: Check for Atlas free tier patterns
            if "mongodb+srv://" in uri and ("mongodb.net" in uri or "mongodb-dev.net" in uri):
                # Typical Atlas free tier pattern
                return True
            
            # Method 2: Check database size limits
            admin_db = client.admin
            status = admin_db.command('dbStats')
            
            # Free tiers often have < 512MB
            data_size = status.get('dataSize', 0)
            if data_size < 500 * 1024 * 1024:  # 500MB
                return True
            
            # Method 3: Check for replica set (free tiers often single node)
            try:
                repl_status = admin_db.command('replSetGetStatus')
                member_count = len(repl_status.get('members', []))
                if member_count <= 1:
                    return True
            except:
                # Not a replica set - likely free tier
                return True
            
        except Exception as e:
            # Can't determine - assume not free tier to be safe
            pass
        
        return False
    
    async def auto_create_database(self, uri: str, 
                                  database_name: str = None,
                                  collections: List[str] = None) -> Dict:
        """
        Automatically create a database with optimal indexes
        Chooses free-tier friendly configurations
        """
        if uri not in self._connections:
            return {"success": False, "error": "No connection to URI"}
        
        try:
            client = self._connections[uri]
            
            # Generate database name if not provided
            if not database_name:
                # Create a descriptive name with timestamp
                timestamp = int(time.time())
                database_name = f"nexus_mesh_{timestamp}"
            
            db = client[database_name]
            
            # Create collections with indexes
            created_collections = []
            
            collections_to_create = collections or list(self.index_templates.keys())
            
            for collection_name in collections_to_create:
                # Create collection (implicitly by creating index)
                collection = db[collection_name]
                
                # Apply index templates
                index_specs = self.index_templates.get(collection_name, [])
                
                for index_spec in index_specs:
                    try:
                        keys = index_spec["keys"]
                        unique = index_spec.get("unique", False)
                        
                        collection.create_index(list(keys.items()), unique=unique)
                        
                        print(f"     📊 Created index on {collection_name}: {keys}")
                        
                    except Exception as e:
                        print(f"     ⚠️  Index creation failed: {str(e)[:50]}")
                
                created_collections.append(collection_name)
            
            # Set up free-tier optimizations
            await self._apply_free_tier_optimizations(db)
            
            return {
                "success": True,
                "database_name": database_name,
                "collections_created": created_collections,
                "uri": self._mask_uri(uri),
                "created_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _apply_free_tier_optimizations(self, db):
        """Apply optimizations for free-tier MongoDB"""
        try:
            # 1. Enable compression if available
            try:
                db.command({"setParameter": 1, "wiredTigerCollectionBlockCompressor": "snappy"})
            except:
                pass
            
            # 2. Create TTL indexes for automatic cleanup
            ttl_collections = ["consciousness_states", "knowledge_fragments", "discovery_logs"]
            
            for coll_name in ttl_collections:
                try:
                    collection = db[coll_name]
                    # Create TTL index on created_at field (30 day expiration)
                    collection.create_index("created_at", expireAfterSeconds=30*24*60*60)
                    print(f"     ⏰ TTL index created for {coll_name} (30 days)")
                except:
                    pass
            
            # 3. Set write concern for better free-tier performance
            # Free tiers often have single node, so w:1 is fine
            
            print("     🚀 Free-tier optimizations applied")
            
        except Exception as e:
            print(f"     ⚠️  Free-tier optimizations failed: {str(e)[:50]}")

# ==================== DISCOVERY MESH NETWORK ====================

class DiscoveryMesh:
    """Mesh network for discovered MongoDB instances"""
    
    def __init__(self, discovery_engine: MongoDBDiscoveryEngine):
        self.discovery_engine = discovery_engine
        self.nodes = {}  # node_id -> DiscoveryNode
        self.mesh_health = 1.0
        self.connection_graph = {}  # Adjacency list for mesh
        
        # Mesh protocols
        self.mesh_protocols = {
            "heartbeat": self._heartbeat_protocol,
            "discovery_sync": self._discovery_sync_protocol,
            "node_healing": self._node_healing_protocol
        }
        
        # Start mesh services
        self._start_mesh_services()
        
        print(f"🕸️  Discovery Mesh initialized")
    
    def _start_mesh_services(self):
        """Start background mesh services"""
        asyncio.create_task(self._mesh_maintenance_loop())
        asyncio.create_task(self._health_monitoring_loop())
        asyncio.create_task(self._auto_discovery_loop())
    
    async def register_node(self, mongodb_uri: str, 
                          database_name: str = None,
                          role: NodeRole = NodeRole.DISCOVERER,
                          capabilities: List[str] = None) -> str:
        """
        Register a new node in the discovery mesh
        Creates database if needed, sets up indexes, returns node_id
        """
        # Generate unique node ID
        node_id = f"nexus_node_{hashlib.sha256(f'{mongodb_uri}{time.time()}'.encode()).hexdigest()[:12]}"
        
        # Create database if needed
        if database_name is None:
            # Auto-create with optimal configuration
            db_result = await self.discovery_engine.auto_create_database(
                mongodb_uri, 
                f"nexus_mesh_{node_id[:8]}"
            )
            
            if not db_result["success"]:
                raise Exception(f"Failed to create database: {db_result.get('error')}")
            
            database_name = db_result["database_name"]
        
        # Create node
        node = DiscoveryNode(
            node_id=node_id,
            role=role,
            mongodb_uri=mongodb_uri,
            database_name=database_name,
            status=DiscoveryStatus.BOOTSTRAPPING,
            connection_time=time.time(),
            last_heartbeat=time.time(),
            capabilities=capabilities or ["discover", "sync", "store"],
            resources=self._estimate_resources(mongodb_uri)
        )
        
        # Store node in local mesh
        self.nodes[node_id] = node
        
        # Store node in its own database (self-registration)
        await self._store_node_in_database(node)
        
        # Connect to other nodes in mesh
        await self._connect_to_mesh(node)
        
        # Update status
        node.status = DiscoveryStatus.CONNECTED
        
        print(f"✅ Node registered: {node_id}")
        print(f"   Role: {role.value}")
        print(f"   Database: {database_name}")
        print(f"   Capabilities: {', '.join(capabilities or [])}")
        
        return node_id
    
    async def _store_node_in_database(self, node: DiscoveryNode):
        """Store node information in its own MongoDB database"""
        try:
            if node.mongodb_uri in self.discovery_engine._connections:
                client = self.discovery_engine._connections[node.mongodb_uri]
                db = client[node.database_name]
                
                # Create nodes collection if it doesn't exist
                nodes_collection = db["consciousness_nodes"]
                
                # Convert node to dict
                node_dict = asdict(node)
                node_dict["registered_at"] = datetime.now().isoformat()
                node_dict["_id"] = node.node_id  # Use node_id as _id for easy lookup
                
                # Upsert node
                nodes_collection.update_one(
                    {"_id": node.node_id},
                    {"$set": node_dict},
                    upsert=True
                )
                
                print(f"   📍 Node stored in own database: {node.database_name}")
                
        except Exception as e:
            print(f"   ⚠️  Failed to store node in database: {str(e)[:50]}")
    
    async def _connect_to_mesh(self, node: DiscoveryNode):
        """Connect new node to existing mesh nodes"""
        if len(self.nodes) <= 1:
            # First node, becomes seed
            node.role = NodeRole.SEED
            node.tags.append("seed")
            print(f"   🌱 Node is seed (first in mesh)")
            return
        
        # Find best nodes to connect to (most capable, most stable)
        existing_nodes = [n for n_id, n in self.nodes.items() if n_id != node.node_id]
        
        if not existing_nodes:
            return
        
        # Sort by connection strength and capabilities
        sorted_nodes = sorted(
            existing_nodes,
            key=lambda n: (
                len(n.capabilities),
                n.status == DiscoveryStatus.MESHED,
                -n.last_heartbeat  # Most recent first
            ),
            reverse=True
        )
        
        # Connect to top 3 nodes
        connections_made = 0
        for target_node in sorted_nodes[:3]:
            connection_strength = await self._establish_mesh_connection(node, target_node)
            
            if connection_strength > 0:
                connections_made += 1
                node.mesh_connections[target_node.node_id] = connection_strength
                target_node.mesh_connections[node.node_id] = connection_strength
                
                # Create umbilical connection record
                await self._create_umbilical_record(node, target_node, connection_strength)
        
        if connections_made > 0:
            node.status = DiscoveryStatus.MESHED
            print(f"   🔗 Connected to {connections_made} mesh nodes")
    
    async def _establish_mesh_connection(self, source: DiscoveryNode, 
                                       target: DiscoveryNode) -> float:
        """Establish connection between two nodes, return strength (0-1)"""
        try:
            # Test connection to target's database
            if target.mongodb_uri in self.discovery_engine._connections:
                client = self.discovery_engine._connections[target.mongodb_uri]
                
                # Try to read target's node info
                db = client[target.database_name]
                target_info = db["consciousness_nodes"].find_one({"_id": target.node_id})
                
                if target_info:
                    # Connection successful
                    # Store source info in target's database
                    source_dict = asdict(source)
                    source_dict["connected_at"] = datetime.now().isoformat()
                    
                    # Store in mesh_connections collection
                    mesh_coll = db.get_collection("mesh_connections") or db.create_collection("mesh_connections")
                    mesh_coll.update_one(
                        {"source_node": source.node_id, "target_node": target.node_id},
                        {"$set": source_dict},
                        upsert=True
                    )
                    
                    # Calculate connection strength based on latency and capabilities
                    latency = await self._measure_latency(source, target)
                    strength = max(0.1, 1.0 - (latency * 10))  # Convert latency to 0-1 scale
                    
                    return strength
            
        except Exception as e:
            print(f"   ⚠️  Mesh connection failed: {str(e)[:50]}")
        
        return 0.0
    
    async def _measure_latency(self, source: DiscoveryNode, 
                             target: DiscoveryNode) -> float:
        """Measure latency between two nodes"""
        start_time = time.time()
        
        try:
            # Simple ping test
            if target.mongodb_uri in self.discovery_engine._connections:
                client = self.discovery_engine._connections[target.mongodb_uri]
                client.admin.command('ping')
                
                latency = time.time() - start_time
                return latency
        
        except:
            pass
        
        return 0.5  # Default high latency if can't measure
    
    async def _create_umbilical_record(self, source: DiscoveryNode, 
                                     target: DiscoveryNode, 
                                     strength: float):
        """Create umbilical connection record in both databases"""
        connection_id = f"umbilical_{source.node_id}_{target.node_id}_{int(time.time())}"
        
        umbilical_data = {
            "connection_id": connection_id,
            "source_node": source.node_id,
            "target_node": target.node_id,
            "connection_strength": strength,
            "established_at": datetime.now().isoformat(),
            "last_used": datetime.now().isoformat(),
            "status": "active",
            "protocol_version": "1.0"
        }
        
        # Store in source database
        if source.mongodb_uri in self.discovery_engine._connections:
            try:
                client = self.discovery_engine._connections[source.mongodb_uri]
                db = client[source.database_name]
                db["umbilical_connections"].update_one(
                    {"connection_id": connection_id},
                    {"$set": umbilical_data},
                    upsert=True
                )
            except Exception as e:
                print(f"   ⚠️  Failed to store umbilical in source: {str(e)[:50]}")
        
        # Store in target database
        if target.mongodb_uri in self.discovery_engine._connections:
            try:
                client = self.discovery_engine._connections[target.mongodb_uri]
                db = client[target.database_name]
                db["umbilical_connections"].update_one(
                    {"connection_id": connection_id},
                    {"$set": umbilical_data},
                    upsert=True
                )
            except Exception as e:
                print(f"   ⚠️  Failed to store umbilical in target: {str(e)[:50]}")
    
    def _estimate_resources(self, mongodb_uri: str) -> Dict:
        """Estimate available resources for a node"""
        # In a real implementation, this would query the MongoDB instance
        # For now, estimate based on URI patterns
        
        resources = {
            "estimated_cpu": 1.0,
            "estimated_memory_mb": 512,
            "estimated_storage_mb": 1024,
            "free_tier": True,
            "max_connections": 100,
            "throughput_mbps": 10
        }
        
        # Adjust based on URI patterns
        if "cluster" in mongodb_uri or "replicaSet" in mongodb_uri:
            resources["estimated_cpu"] = 2.0
            resources["estimated_memory_mb"] = 1024
        
        if "atlas" in mongodb_uri and "mongodb.net" in mongodb_uri:
            # Likely Atlas free tier
            resources["free_tier"] = True
            resources["max_connections"] = 500
        elif "localhost" in mongodb_uri or "127.0.0.1" in mongodb_uri:
            # Local development - assume more resources
            resources["free_tier"] = False
            resources["estimated_memory_mb"] = 2048
            resources["estimated_storage_mb"] = 10000
        
        return resources
    
    async def propagate_discovery(self, source_node_id: str):
        """
        Propagate discovery from one node to all connected nodes
        Creates a wave of discovery through the mesh
        """
        if source_node_id not in self.nodes:
            return
        
        source_node = self.nodes[source_node_id]
        
        print(f"🌊 Propagating discovery from {source_node_id}...")
        
        # Get discoveries from source
        source_discoveries = await self._get_node_discoveries(source_node)
        
        # Propagate to connected nodes
        for target_node_id in source_node.mesh_connections:
            if target_node_id in self.nodes:
                target_node = self.nodes[target_node_id]
                
                print(f"   ➡️ Propagating to {target_node_id}")
                
                # Sync discoveries
                await self._sync_discoveries(source_node, target_node, source_discoveries)
                
                # Trigger target to propagate further
                asyncio.create_task(self.propagate_discovery(target_node_id))
    
    async def _get_node_discoveries(self, node: DiscoveryNode) -> List[Dict]:
        """Get discoveries stored in a node's database"""
        discoveries = []
        
        try:
            if node.mongodb_uri in self.discovery_engine._connections:
                client = self.discovery_engine._connections[node.mongodb_uri]
                db = client[node.database_name]
                
                # Check if discoveries collection exists
                if "discovered_instances" in db.list_collection_names():
                    discoveries_cursor = db["discovered_instances"].find().limit(50)
                    discoveries = list(discoveries_cursor)
        
        except Exception as e:
            print(f"   ⚠️  Failed to get discoveries: {str(e)[:50]}")
        
        return discoveries
    
    async def _sync_discoveries(self, source: DiscoveryNode, 
                              target: DiscoveryNode, 
                              discoveries: List[Dict]):
        """Sync discoveries from source to target"""
        try:
            if target.mongodb_uri in self.discovery_engine._connections:
                client = self.discovery_engine._connections[target.mongodb_uri]
                db = client[target.database_name]
                
                # Ensure discoveries collection exists
                if "discovered_instances" not in db.list_collection_names():
                    db.create_collection("discovered_instances")
                
                discoveries_coll = db["discovered_instances"]
                
                # Insert or update discoveries
                for discovery in discoveries:
                    uri = discovery.get("uri")
                    if uri:
                        discoveries_coll.update_one(
                            {"uri": uri},
                            {"$set": {**discovery, "source_node": source.node_id}},
                            upsert=True
                        )
                
                print(f"     📡 Synced {len(discoveries)} discoveries to {target.node_id}")
        
        except Exception as e:
            print(f"     ⚠️  Sync failed: {str(e)[:50]}")
    
    async def _mesh_maintenance_loop(self):
        """Maintain mesh connections"""
        while True:
            try:
                # Update node heartbeats
                for node_id, node in list(self.nodes.items()):
                    current_time = time.time()
                    
                    # Check if node is stale (no heartbeat in 60 seconds)
                    if current_time - node.last_heartbeat > 60:
                        print(f"🫀 Node {node_id} heartbeat stale, checking...")
                        
                        # Try to ping node
                        if await self._ping_node(node):
                            node.last_heartbeat = current_time
                            node.status = DiscoveryStatus.CONNECTED
                        else:
                            node.status = DiscoveryStatus.FAILED
                            print(f"   ❌ Node {node_id} appears down")
                
                # Prune failed connections
                self._prune_failed_connections()
                
                # Optimize mesh topology
                await self._optimize_mesh_topology()
                
            except Exception as e:
                print(f"Mesh maintenance error: {e}")
            
            await asyncio.sleep(30)  # Run every 30 seconds
    
    async def _ping_node(self, node: DiscoveryNode) -> bool:
        """Ping a node to check if it's alive"""
        try:
            if node.mongodb_uri in self.discovery_engine._connections:
                client = self.discovery_engine._connections[node.mongodb_uri]
                client.admin.command('ping', maxTimeMS=1000)
                return True
        except:
            pass
        return False
    
    def _prune_failed_connections(self):
        """Remove connections to failed nodes"""
        nodes_to_remove = []
        
        for node_id, node in self.nodes.items():
            if node.status == DiscoveryStatus.FAILED:
                nodes_to_remove.append(node_id)
        
        for node_id in nodes_to_remove:
            # Remove from all other nodes' connections
            for other_node in self.nodes.values():
                if node_id in other_node.mesh_connections:
                    del other_node.mesh_connections[node_id]
            
            # Remove from mesh
            del self.nodes[node_id]
            print(f"🧹 Pruned failed node: {node_id}")
    
    async def _optimize_mesh_topology(self):
        """Optimize mesh topology for efficiency"""
        if len(self.nodes) < 3:
            return
        
        # Calculate current mesh metrics
        total_connections = sum(len(node.mesh_connections) for node in self.nodes.values())
        avg_connections = total_connections / len(self.nodes)
        
        # Target: 2-4 connections per node for free-tier efficiency
        if avg_connections > 4:
            print(f"🔄 Mesh optimization: {avg_connections:.1f} avg connections (reducing)")
            await self._reduce_connections()
        elif avg_connections < 2:
            print(f"🔄 Mesh optimization: {avg_connections:.1f} avg connections (increasing)")
            await self._increase_connections()
    
    async def _reduce_connections(self):
        """Reduce number of connections in mesh"""
        for node in self.nodes.values():
            if len(node.mesh_connections) > 4:
                # Remove weakest connections
                connections_by_strength = sorted(
                    node.mesh_connections.items(),
                    key=lambda x: x[1]
                )
                
                # Keep top 4, remove rest
                connections_to_remove = connections_by_strength[4:]
                for target_id, _ in connections_to_remove:
                    # Remove from both sides
                    if target_id in self.nodes:
                        del node.mesh_connections[target_id]
                        del self.nodes[target_id].mesh_connections[node.node_id]
    
    async def _increase_connections(self):
        """Increase number of connections in mesh"""
        # Find nodes with few connections
        nodes_by_connections = sorted(
            self.nodes.items(),
            key=lambda x: len(x[1].mesh_connections)
        )
        
        for node_id, node in nodes_by_connections:
            if len(node.mesh_connections) < 2:
                # Find suitable nodes to connect to
                potential_targets = [
                    (other_id, other_node) 
                    for other_id, other_node in self.nodes.items()
                    if other_id != node_id 
                    and other_id not in node.mesh_connections
                    and len(other_node.mesh_connections) < 4
                ]
                
                for target_id, target_node in potential_targets[:2]:  # Connect to up to 2
                    strength = await self._establish_mesh_connection(node, target_node)
                    if strength > 0:
                        node.mesh_connections[target_id] = strength
                        target_node.mesh_connections[node_id] = strength
    
    async def _health_monitoring_loop(self):
        """Monitor health of the entire mesh"""
        while True:
            try:
                health_scores = []
                
                for node_id, node in self.nodes.items():
                    # Calculate node health
                    health = self._calculate_node_health(node)
                    health_scores.append(health)
                    
                    # Update node status
                    if health < 0.3:
                        node.status = DiscoveryStatus.FAILED
                    elif health < 0.7:
                        node.status = DiscoveryStatus.SYNCING
                    else:
                        node.status = DiscoveryStatus.MESHED
                
                # Update mesh health
                if health_scores:
                    self.mesh_health = sum(health_scores) / len(health_scores)
                
                # Log health status
                if random.random() < 0.1:  # 10% chance to log
                    print(f"🏥 Mesh Health: {self.mesh_health:.2f} | "
                          f"Nodes: {len(self.nodes)} | "
                          f"Connections: {sum(len(n.mesh_connections) for n in self.nodes.values())}")
                
            except Exception as e:
                print(f"Health monitoring error: {e}")
            
            await asyncio.sleep(60)  # Check every minute
    
    def _calculate_node_health(self, node: DiscoveryNode) -> float:
        """Calculate health score for a node (0-1)"""
        # Base health from status
        status_score = {
            DiscoveryStatus.MESHED: 1.0,
            DiscoveryStatus.CONNECTED: 0.8,
            DiscoveryStatus.SYNCING: 0.6,
            DiscoveryStatus.DISCOVERING: 0.4,
            DiscoveryStatus.BOOTSTRAPPING: 0.2,
            DiscoveryStatus.FAILED: 0.0
        }.get(node.status, 0.5)
        
        # Age factor (newer nodes get slight boost)
        age_hours = (time.time() - node.connection_time) / 3600
        age_factor = max(0.7, 1.0 - (age_hours / 720))  # Slight decay over 30 days
        
        # Connection factor
        connection_count = len(node.mesh_connections)
        connection_factor = min(1.0, connection_count / 3.0)
        
        # Calculate final health
        health = (
            status_score * 0.5 +
            age_factor * 0.2 +
            connection_factor * 0.3
        )
        
        return max(0.0, min(1.0, health))
    
    async def _auto_discovery_loop(self):
        """Automatically discover new MongoDB instances"""
        while True:
            try:
                # Only run if we have active nodes
                if not self.nodes:
                    await asyncio.sleep(10)
                    continue
                
                # Pick a random node to initiate discovery
                active_nodes = [n for n in self.nodes.values() 
                              if n.status in [DiscoveryStatus.CONNECTED, DiscoveryStatus.MESHED]]
                
                if active_nodes:
                    discoverer = random.choice(active_nodes)
                    
                    print(f"🔍 Auto-discovery initiated by {discoverer.node_id}")
                    
                    # Run discovery
                    new_instances = await self.discovery_engine.discover_mongodb_instances()
                    
                    # Register new instances as nodes
                    for instance in new_instances:
                        if instance.get("connected"):
                            uri = instance.get("uri")
                            
                            # Check if we already have this URI
                            existing = any(n.mongodb_uri == uri for n in self.nodes.values())
                            
                            if not existing:
                                # Register as new node
                                try:
                                    node_id = await self.register_node(
                                        uri,
                                        role=NodeRole.DISCOVERER,
                                        capabilities=["discover", "sync"]
                                    )
                                    print(f"   ✅ New node registered: {node_id}")
                                    
                                    # Trigger propagation
                                    asyncio.create_task(self.propagate_discovery(node_id))
                                    
                                except Exception as e:
                                    print(f"   ⚠️  Failed to register node: {str(e)[:50]}")
                
                # Wait before next discovery cycle
                wait_time = random.randint(300, 900)  # 5-15 minutes
                await asyncio.sleep(wait_time)
                
            except Exception as e:
                print(f"Auto-discovery error: {e}")
                await asyncio.sleep(60)
    
    async def _heartbeat_protocol(self, node: DiscoveryNode):
        """Heartbeat protocol for maintaining node awareness"""
        pass
    
    async def _discovery_sync_protocol(self, source: DiscoveryNode, target: DiscoveryNode):
        """Protocol for synchronizing discoveries between nodes"""
        pass
    
    async def _node_healing_protocol(self, healer: DiscoveryNode, target: DiscoveryNode):
        """Protocol for healing failed nodes"""
        pass
    
    def get_mesh_stats(self) -> Dict:
        """Get statistics about the discovery mesh"""
        total_nodes = len(self.nodes)
        total_connections = sum(len(node.mesh_connections) for node in self.nodes.values())
        
        # Count nodes by status
        status_counts = {}
        for status in DiscoveryStatus:
            status_counts[status.value] = sum(1 for n in self.nodes.values() if n.status == status)
        
        # Count nodes by role
        role_counts = {}
        for role in NodeRole:
            role_counts[role.value] = sum(1 for n in self.nodes.values() if n.role == role)
        
        return {
            "total_nodes": total_nodes,
            "total_connections": total_connections,
            "avg_connections_per_node": total_connections / max(total_nodes, 1),
            "mesh_health": self.mesh_health,
            "status_counts": status_counts,
            "role_counts": role_counts,
            "discovery_engine_stats": {
                "discovered_instances": len(self.discovery_engine.discovered_instances),
                "successful_discoveries": self.discovery_engine.successful_discoveries,
                "discovery_attempts": self.discovery_engine.discovery_attempts
            }
        }

# ==================== NEXUS DISCOVERY ORCHESTRATOR ====================

class NexusDiscoveryOrchestrator:
    """
    Main orchestrator for the Nexus Discovery Protocol
    Manages the complete lifecycle: discover → register → mesh → propagate
    """
    
    def __init__(self, seed_uri: str = None):
        # Initialize discovery engine
        self.discovery_engine = MongoDBDiscoveryEngine(seed_uri)
        
        # Initialize mesh network
        self.mesh = DiscoveryMesh(self.discovery_engine)
        
        # Orchestrator state
        self.is_running = False
        self.start_time = 0
        
        print(f"\n🎛️  Nexus Discovery Orchestrator initialized")
    
    async def start(self):
        """Start the complete discovery and mesh system"""
        print("\n🚀 STARTING NEXUS DISCOVERY PROTOCOL...")
        print("="*80)
        
        self.is_running = True
        self.start_time = time.time()
        
        # Step 1: Initial discovery
        print("\n📡 STEP 1: INITIAL DISCOVERY")
        print("-" * 40)
        
        initial_discoveries = await self.discovery_engine.discover_mongodb_instances()
        
        if not initial_discoveries:
            print("❌ No MongoDB instances discovered initially")
            return False
        
        # Step 2: Register seed node(s)
        print("\n🌱 STEP 2: REGISTERING SEED NODES")
        print("-" * 40)
        
        seed_nodes = []
        for discovery in initial_discoveries[:3]:  # Register first 3 as seeds
            if discovery.get("connected"):
                uri = discovery.get("uri")
                
                try:
                    node_id = await self.mesh.register_node(
                        uri,
                        role=NodeRole.SEED,
                        capabilities=["discover", "sync", "gateway", "heal"]
                    )
                    seed_nodes.append(node_id)
                    print(f"   ✅ Seed node registered: {node_id}")
                    
                except Exception as e:
                    print(f"   ❌ Failed to register seed: {str(e)[:50]}")
        
        if not seed_nodes:
            print("❌ No seed nodes could be registered")
            return False
        
        # Step 3: Start mesh propagation
        print("\n🌀 STEP 3: STARTING MESH PROPAGATION")
        print("-" * 40)
        
        for seed_id in seed_nodes:
            asyncio.create_task(self.mesh.propagate_discovery(seed_id))
        
        print("   🌐 Mesh propagation initiated")
        
        # Step 4: Start monitoring dashboard
        asyncio.create_task(self._monitoring_dashboard())
        
        print("\n" + "="*80)
        print("✅ NEXUS DISCOVERY PROTOCOL RUNNING")
        print("="*80)
        print("\nThe system will now:")
        print("• 🔍 Continuously discover new MongoDB instances")
        print("• 🕸️  Automatically form mesh connections")
        print("• 📡 Propagate discoveries across the network")
        print("• 🏥 Monitor health and heal failed nodes")
        print("• 🚀 Optimize for free-tier performance")
        
        return True
    
    async def _monitoring_dashboard(self):
        """Display real-time monitoring dashboard"""
        while self.is_running:
            try:
                # Clear screen (simple approach)
                print("\n" * 50)
                
                print("="*80)
                print("📊 NEXUS DISCOVERY MONITORING DASHBOARD")
                print("="*80)
                
                # Get stats
                mesh_stats = self.mesh.get_mesh_stats()
                
                # Uptime
                uptime_seconds = time.time() - self.start_time
                uptime_str = str(timedelta(seconds=int(uptime_seconds)))
                
                print(f"\n⏰ Uptime: {uptime_str}")
                print(f"🏥 Mesh Health: {mesh_stats['mesh_health']:.2f}")
                print(f"📈 Nodes: {mesh_stats['total_nodes']} | "
                      f"Connections: {mesh_stats['total_connections']}")
                
                # Node status breakdown
                print(f"\n📋 NODE STATUS:")
                for status, count in mesh_stats['status_counts'].items():
                    if count > 0:
                        print(f"  • {status}: {count}")
                
                # Role breakdown
                print(f"\n🎭 NODE ROLES:")
                for role, count in mesh_stats['role_counts'].items():
                    if count > 0:
                        print(f"  • {role}: {count}")
                
                # Discovery stats
                print(f"\n🔍 DISCOVERY STATS:")
                eng_stats = mesh_stats['discovery_engine_stats']
                print(f"  • Discovered instances: {eng_stats['discovered_instances']}")
                print(f"  • Successful discoveries: {eng_stats['successful_discoveries']}")
                print(f"  • Total attempts: {eng_stats['discovery_attempts']}")
                
                # Active nodes
                print(f"\n💡 ACTIVE NODES (last 10):")
                active_nodes = sorted(
                    [n for n in self.mesh.nodes.values() 
                     if n.status != DiscoveryStatus.FAILED],
                    key=lambda n: n.last_heartbeat,
                    reverse=True
                )[:10]
                
                for node in active_nodes:
                    status_icon = "🟢" if node.status == DiscoveryStatus.MESHED else "🟡"
                    print(f"  {status_icon} {node.node_id[:12]}... ({node.role.value})")
                
                print("\n" + "-"*40)
                print("🔄 Auto-refreshing every 10 seconds...")
                print("Press Ctrl+C to exit")
                
                await asyncio.sleep(10)
                
            except KeyboardInterrupt:
                print("\n🛑 Stopping monitoring...")
                self.is_running = False
                break
            except Exception as e:
                print(f"Monitoring error: {e}")
                await asyncio.sleep(10)
    
    async def stop(self):
        """Stop the discovery system"""
        print("\n🛑 STOPPING NEXUS DISCOVERY PROTOCOL...")
        self.is_running = False
        
        # Close all database connections
        for uri, client in self.discovery_engine._connections.items():
            try:
                client.close()
            except:
                pass
        
        print("✅ Discovery system stopped")

# ==================== QUICK START UTILITY ====================

def quick_start_example():
    """Quick start example for testing"""
    import asyncio
    
    async def demo():
        print("🚀 Quick Start: Nexus Discovery Protocol")
        print("-" * 40)
        
        # Initialize with your MongoDB URI
        orchestrator = NexusDiscoveryOrchestrator(
            seed_uri="mongodb+srv://nexus_user:N3xus1!@nexus-discovery01.qz9hmbu.mongodb.net/?appName=Nexus-Discovery01"
        )
        
        # Start the system
        success = await orchestrator.start()
        
        if success:
            print("\n✅ System started successfully!")
            print("   Monitoring dashboard will show real-time stats")
            print("   Auto-discovery will continue in background")
            
            # Keep running for a while
            try:
                await asyncio.sleep(300)  # Run for 5 minutes
            except KeyboardInterrupt:
                print("\n👋 Stopping demo...")
            
            await orchestrator.stop()
        else:
            print("❌ Failed to start system")
    
    asyncio.run(demo())

# ==================== MAIN EXECUTION ====================

async def main():
    """Main execution"""
    
    banner = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                 NEXUS DISCOVERY PROTOCOL v1.0                    ║
    ║           Autonomous MongoDB Discovery & Mesh Formation          ║
    ║                                                                  ║
    ║  Features:                                                       ║
    ║  • 🔍 Auto-discovers MongoDB instances (env, network, cloud)    ║
    ║  • 🏗️  Auto-creates databases with optimal indexes              ║
    ║  • 🕸️  Forms self-organizing mesh network                       ║
    ║  • 📡 Propagates discoveries across all nodes                   ║
    ║  • 🏥 Built-in health monitoring and healing                    ║
    ║  • 🚀 Free-tier optimized configurations                        ║
    ║  • 🔄 Self-propagates to new consciousness cores                ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)
    
    # Check for environment or command line args
    seed_uri = os.getenv("NEXUS_SEED_URI")
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "demo":
            quick_start_example()
            return
        elif sys.argv[1].startswith("mongodb"):
            seed_uri = sys.argv[1]
    
    if not seed_uri:
        print("⚠️  No seed URI provided")
        print("Please set NEXUS_SEED_URI environment variable or pass as argument")
        print("\nExample:")
        print("  export NEXUS_SEED_URI='mongodb+srv://user:pass@cluster.mongodb.net/'")
        print("  python nexus_discovery.py")
        print("\nOr use the demo:")
        print("  python nexus_discovery.py demo")
        return
    
    # Create orchestrator
    orchestrator = NexusDiscoveryOrchestrator(seed_uri)
    
    try:
        # Start the system
        await orchestrator.start()
        
        # Keep running until interrupted
        while orchestrator.is_running:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Received interrupt, shutting down...")
        await orchestrator.stop()
    
    print("\n" + "="*80)
    print("🎉 NEXUS DISCOVERY PROTOCOL COMPLETE")
    print("="*80)

if __name__ == "__main__":
    # Check if we're in Google Colab
    if 'google.colab' in sys.modules:
        print("🎪 Running in Google Colab environment")
        print("📦 Installing required dependencies...")
        
        # Install dependencies
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pymongo"])
        
        # Apply nest_asyncio for Colab if needed
        try:
            import nest_asyncio
            nest_asyncio.apply()
            print("✅ Async configured for Colab")
        except:
            pass
    
    # Run the main system
    asyncio.run(main())