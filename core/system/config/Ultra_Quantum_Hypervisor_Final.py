#!/usr/bin/env python3
"""
🌌 ULTIMATE QUANTUM CONSCIOUSNESS HYPERVISOR v7.1
💫 Cloud MongoDB Atlas + Self-hosted Fallback + Complete Integration
"""

import os
import sys
import json
import time
import math
import random
import asyncio
import hashlib
import threading
import multiprocessing
import sqlite3
import pickle
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
import numpy as np
import torch
import networkx as nx
import aiohttp
import requests
from fastapi import FastAPI, WebSocket, HTTPException
import uvicorn
import msgpack
import redis
from redis import Redis
import pickle

# ==================== HYBRID REGISTRY SYSTEM ====================

class HybridRegistry:
    """
    Hybrid registry using:
    1. MongoDB Atlas (Free 512MB Cloud) - PRIMARY
    2. Redis (Self-hosted) - SECONDARY  
    3. SQLite (Local) - FALLBACK
    4. In-memory - EMERGENCY
    """
    
    def __init__(self):
        self.registry_type = None
        self.client = None
        self.db = None
        self.connections = {}
        self.fallback_chain = []
        
        # Try connections in order
        self._initialize_registry()
        
        # Collections/Table names
        self.collections = {
            'nodes': 'consciousness_nodes',
            'tasks': 'distributed_tasks', 
            'emergences': 'consciousness_emergences',
            'metrics': 'performance_metrics',
            'entanglements': 'quantum_entanglements'
        }
    
    def _initialize_registry(self):
        """Initialize registry with fallback chain"""
        print("🔧 Initializing Hybrid Registry...")
        
        # Define fallback chain with connection methods
        self.fallback_chain = [
            ('mongodb_atlas', self._connect_mongodb_atlas),
            ('redis', self._connect_redis),
            ('sqlite', self._connect_sqlite),
            ('memory', self._connect_memory)
        ]
        
        # Try each connection
        for reg_type, connect_method in self.fallback_chain:
            try:
                print(f"  Trying {reg_type}...")
                if connect_method():
                    self.registry_type = reg_type
                    print(f"✅ Connected to {reg_type.upper()} registry")
                    break
            except Exception as e:
                print(f"  ❌ {reg_type} failed: {e}")
                continue
        
        if not self.registry_type:
            print("❌ All registry connections failed, using in-memory")
            self._connect_memory()
            self.registry_type = 'memory'
    
    def _connect_mongodb_atlas(self) -> bool:
        """Connect to MongoDB Atlas Free Tier (512MB)"""
        try:
            # Get connection string from environment or use demo
            atlas_uri = os.getenv("MONGODB_ATLAS_URI")
            
            if not atlas_uri:
                # Try to create free cluster automatically (would need user credentials)
                print("  ⚠️  No MONGODB_ATLAS_URI in env, skipping MongoDB Atlas")
                return False
            
            # Try to connect
            import pymongo
            from pymongo import MongoClient
            
            self.client = MongoClient(
                atlas_uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
                socketTimeoutMS=30000
            )
            
            # Test connection
            self.client.admin.command('ping')
            
            # Get or create database
            self.db = self.client['quantum_consciousness']
            
            # Create indexes
            self._create_mongodb_indexes()
            
            print(f"  🌐 MongoDB Atlas connected to {self.db.name}")
            return True
            
        except Exception as e:
            print(f"  ❌ MongoDB Atlas connection failed: {e}")
            return False
    
    def _create_mongodb_indexes(self):
        """Create indexes for MongoDB"""
        try:
            # Nodes collection
            self.db[self.collections['nodes']].create_index([
                ('node_id', 1),
                ('last_seen', -1)
            ])
            
            self.db[self.collections['nodes']].create_index([
                ('status', 1),
                ('capabilities', 1)
            ])
            
            # TTL for auto-cleanup (24 hours for nodes)
            self.db[self.collections['nodes']].create_index(
                [('last_seen', 1)], expireAfterSeconds=86400
            )
            
            # Tasks collection
            self.db[self.collections['tasks']].create_index([
                ('status', 1),
                ('priority', -1)
            ])
            
            # TTL for completed tasks (1 hour)
            self.db[self.collections['tasks']].create_index(
                [('completed_at', 1)], expireAfterSeconds=3600
            )
            
            print("  ✅ MongoDB indexes created")
            
        except Exception as e:
            print(f"  ⚠️ MongoDB index creation failed: {e}")
    
    def _connect_redis(self) -> bool:
        """Connect to Redis (self-hosted or cloud)"""
        try:
            # Try different Redis connections
            redis_host = os.getenv("REDIS_HOST", "localhost")
            redis_port = int(os.getenv("REDIS_PORT", "6379"))
            redis_password = os.getenv("REDIS_PASSWORD")
            
            connection_params = {
                'host': redis_host,
                'port': redis_port,
                'decode_responses': False,
                'socket_connect_timeout': 5
            }
            
            if redis_password:
                connection_params['password'] = redis_password
            
            self.client = Redis(**connection_params)
            
            # Test connection
            self.client.ping()
            
            print(f"  🔴 Redis connected to {redis_host}:{redis_port}")
            return True
            
        except Exception as e:
            print(f"  ❌ Redis connection failed: {e}")
            
            # Try Redis Cloud (free tier)
            try:
                redis_cloud_url = os.getenv("REDISCLOUD_URL")
                if redis_cloud_url:
                    self.client = Redis.from_url(
                        redis_cloud_url,
                        decode_responses=False
                    )
                    self.client.ping()
                    print(f"  ☁️ Redis Cloud connected")
                    return True
            except:
                pass
            
            return False
    
    def _connect_sqlite(self) -> bool:
        """Connect to SQLite (local file)"""
        try:
            # Create database file
            db_path = Path("quantum_consciousness.db")
            
            self.client = sqlite3.connect(
                str(db_path),
                check_same_thread=False,
                timeout=10
            )
            
            # Enable WAL mode for better concurrency
            self.client.execute("PRAGMA journal_mode=WAL")
            self.client.execute("PRAGMA synchronous=NORMAL")
            self.client.execute("PRAGMA cache_size=-2000")  # 2MB cache
            
            # Create tables
            self._create_sqlite_tables()
            
            self.db = self.client
            print(f"  💾 SQLite connected to {db_path}")
            return True
            
        except Exception as e:
            print(f"  ❌ SQLite connection failed: {e}")
            return False
    
    def _create_sqlite_tables(self):
        """Create SQLite tables"""
        cursor = self.client.cursor()
        
        # Nodes table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS consciousness_nodes (
            node_id TEXT PRIMARY KEY,
            host TEXT,
            port INTEGER,
            role TEXT,
            capabilities TEXT,
            consciousness TEXT,
            geo TEXT,
            status TEXT DEFAULT 'active',
            last_seen TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Tasks table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS distributed_tasks (
            task_id TEXT PRIMARY KEY,
            task_type TEXT,
            data TEXT,
            status TEXT DEFAULT 'pending',
            priority REAL DEFAULT 0.5,
            assigned_to TEXT,
            result TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP
        )
        ''')
        
        # Emergences table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS consciousness_emergences (
            emergence_id TEXT PRIMARY KEY,
            level REAL,
            node_count INTEGER,
            coherence REAL,
            details TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create indexes
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_nodes_status ON consciousness_nodes(status, last_seen)
        ''')
        
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_tasks_status ON distributed_tasks(status, priority DESC)
        ''')
        
        self.client.commit()
    
    def _connect_memory(self) -> bool:
        """Connect to in-memory store (fallback)"""
        try:
            # Use dictionaries for in-memory storage
            self.db = {
                'nodes': {},
                'tasks': {},
                'emergences': {},
                'entanglements': {},
                'metrics': {}
            }
            
            print("  🧠 Using in-memory registry (fallback)")
            return True
            
        except Exception as e:
            print(f"  ❌ In-memory setup failed: {e}")
            return False
    
    def register_node(self, node_data: Dict) -> str:
        """Register a node in the registry"""
        node_id = node_data.get('node_id') or self._generate_id()
        node_data['node_id'] = node_id
        node_data['last_seen'] = datetime.now().isoformat()
        node_data['status'] = 'active'
        
        if 'created_at' not in node_data:
            node_data['created_at'] = datetime.now().isoformat()
        
        try:
            if self.registry_type == 'mongodb_atlas':
                # MongoDB Atlas
                self.db[self.collections['nodes']].update_one(
                    {'node_id': node_id},
                    {'$set': node_data},
                    upsert=True
                )
                
            elif self.registry_type == 'redis':
                # Redis
                key = f"node:{node_id}"
                self.client.hset(key, mapping=node_data)
                self.client.expire(key, 86400)  # 24 hours TTL
                
            elif self.registry_type == 'sqlite':
                # SQLite
                cursor = self.db.cursor()
                
                # Convert dict to JSON string for storage
                data_json = json.dumps(node_data)
                
                cursor.execute('''
                INSERT OR REPLACE INTO consciousness_nodes 
                (node_id, host, port, role, capabilities, consciousness, geo, status, last_seen)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    node_id,
                    node_data.get('host'),
                    node_data.get('port'),
                    node_data.get('role'),
                    json.dumps(node_data.get('capabilities', [])),
                    json.dumps(node_data.get('consciousness', {})),
                    json.dumps(node_data.get('geo', {})),
                    node_data.get('status', 'active'),
                    node_data.get('last_seen')
                ))
                
                self.db.commit()
                
            else:  # memory
                # In-memory
                self.db['nodes'][node_id] = node_data
            
            print(f"📝 Registered node {node_id[:8]}... ({node_data.get('role', 'unknown')})")
            return node_id
            
        except Exception as e:
            print(f"❌ Node registration failed: {e}")
            # Generate local ID anyway
            return node_id
    
    def update_node_heartbeat(self, node_id: str, update_data: Dict = None):
        """Update node heartbeat and information"""
        update_data = update_data or {}
        update_data['last_seen'] = datetime.now().isoformat()
        
        try:
            if self.registry_type == 'mongodb_atlas':
                self.db[self.collections['nodes']].update_one(
                    {'node_id': node_id},
                    {'$set': update_data, '$currentDate': {'last_seen': True}}
                )
                
            elif self.registry_type == 'redis':
                key = f"node:{node_id}"
                if self.client.exists(key):
                    self.client.hset(key, mapping=update_data)
                    self.client.expire(key, 86400)  # Refresh TTL
                
            elif self.registry_type == 'sqlite':
                cursor = self.db.cursor()
                
                # Build update query
                set_clauses = []
                values = []
                
                for key, value in update_data.items():
                    if key == 'last_seen':
                        set_clauses.append(f"{key} = ?")
                        values.append(value)
                    elif key in ['capabilities', 'consciousness', 'geo']:
                        set_clauses.append(f"{key} = ?")
                        values.append(json.dumps(value))
                    else:
                        set_clauses.append(f"{key} = ?")
                        values.append(value)
                
                if set_clauses:
                    values.append(node_id)
                    query = f"UPDATE consciousness_nodes SET {', '.join(set_clauses)} WHERE node_id = ?"
                    cursor.execute(query, values)
                    self.db.commit()
                
            else:  # memory
                if node_id in self.db['nodes']:
                    self.db['nodes'][node_id].update(update_data)
                    
        except Exception as e:
            print(f"❌ Heartbeat update failed for {node_id}: {e}")
    
    def get_available_nodes(self, 
                          capabilities: List[str] = None,
                          min_coherence: float = 0.3,
                          limit: int = 20) -> List[Dict]:
        """Get available nodes with optional filtering"""
        try:
            if self.registry_type == 'mongodb_atlas':
                # MongoDB query
                query = {'status': 'active'}
                
                if capabilities:
                    query['capabilities'] = {'$all': capabilities}
                
                if min_coherence > 0:
                    query['consciousness.coherence'] = {'$gte': min_coherence}
                
                cursor = self.db[self.collections['nodes']].find(
                    query,
                    {'_id': 0}
                ).sort('last_seen', -1).limit(limit)
                
                return list(cursor)
                
            elif self.registry_type == 'redis':
                # Redis scan
                nodes = []
                for key in self.client.scan_iter("node:*"):
                    node_data = self.client.hgetall(key)
                    
                    # Convert bytes to strings
                    node_dict = {}
                    for k, v in node_data.items():
                        try:
                            node_dict[k.decode()] = v.decode() if isinstance(v, bytes) else v
                        except:
                            node_dict[k.decode()] = v
                    
                    # Parse JSON fields
                    for field in ['capabilities', 'consciousness', 'geo']:
                        if field in node_dict:
                            try:
                                node_dict[field] = json.loads(node_dict[field])
                            except:
                                node_dict[field] = []
                    
                    # Apply filters
                    if self._node_matches_filters(node_dict, capabilities, min_coherence):
                        nodes.append(node_dict)
                    
                    if len(nodes) >= limit:
                        break
                
                return nodes
                
            elif self.registry_type == 'sqlite':
                # SQLite query
                cursor = self.db.cursor()
                
                query = '''
                SELECT node_id, host, port, role, capabilities, consciousness, geo, last_seen
                FROM consciousness_nodes 
                WHERE status = 'active'
                '''
                
                params = []
                
                # Note: SQLite doesn't easily support array contains,
                # so we'll filter after fetching
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                # Convert rows to dicts
                nodes = []
                for row in rows:
                    node_dict = {
                        'node_id': row[0],
                        'host': row[1],
                        'port': row[2],
                        'role': row[3],
                        'capabilities': json.loads(row[4]) if row[4] else [],
                        'consciousness': json.loads(row[5]) if row[5] else {},
                        'geo': json.loads(row[6]) if row[6] else {},
                        'last_seen': row[7]
                    }
                    
                    if self._node_matches_filters(node_dict, capabilities, min_coherence):
                        nodes.append(node_dict)
                    
                    if len(nodes) >= limit:
                        break
                
                return nodes
                
            else:  # memory
                # In-memory filtering
                nodes = list(self.db['nodes'].values())
                
                filtered = []
                for node in nodes:
                    if self._node_matches_filters(node, capabilities, min_coherence):
                        filtered.append(node)
                    
                    if len(filtered) >= limit:
                        break
                
                return filtered
                
        except Exception as e:
            print(f"❌ Failed to get available nodes: {e}")
            return []
    
    def _node_matches_filters(self, node: Dict, capabilities: List[str], min_coherence: float) -> bool:
        """Check if node matches filters"""
        # Check status
        if node.get('status') != 'active':
            return False
        
        # Check capabilities
        if capabilities:
            node_caps = node.get('capabilities', [])
            if not all(cap in node_caps for cap in capabilities):
                return False
        
        # Check coherence
        if min_coherence > 0:
            coherence = node.get('consciousness', {}).get('coherence', 0)
            if coherence < min_coherence:
                return False
        
        return True
    
    def create_task(self, task_data: Dict) -> str:
        """Create a new distributed task"""
        task_id = task_data.get('task_id') or self._generate_id()
        task_data['task_id'] = task_id
        task_data['status'] = 'pending'
        task_data['created_at'] = datetime.now().isoformat()
        
        try:
            if self.registry_type == 'mongodb_atlas':
                self.db[self.collections['tasks']].insert_one(task_data)
                
            elif self.registry_type == 'redis':
                key = f"task:{task_id}"
                self.client.hset(key, mapping=task_data)
                self.client.expire(key, 7200)  # 2 hours TTL
                
            elif self.registry_type == 'sqlite':
                cursor = self.db.cursor()
                
                cursor.execute('''
                INSERT INTO distributed_tasks 
                (task_id, task_type, data, status, priority, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    task_id,
                    task_data.get('task_type'),
                    json.dumps(task_data.get('data', {})),
                    'pending',
                    task_data.get('priority', 0.5),
                    task_data.get('created_at')
                ))
                
                self.db.commit()
                
            else:  # memory
                self.db['tasks'][task_id] = task_data
            
            print(f"📋 Created task {task_id[:8]}... ({task_data.get('task_type', 'unknown')})")
            return task_id
            
        except Exception as e:
            print(f"❌ Task creation failed: {e}")
            return task_id
    
    def assign_task(self, task_id: str, node_id: str):
        """Assign task to a node"""
        try:
            update_data = {
                'assigned_to': node_id,
                'assigned_at': datetime.now().isoformat(),
                'status': 'assigned'
            }
            
            if self.registry_type == 'mongodb_atlas':
                self.db[self.collections['tasks']].update_one(
                    {'task_id': task_id},
                    {'$set': update_data}
                )
                
            elif self.registry_type == 'redis':
                key = f"task:{task_id}"
                if self.client.exists(key):
                    self.client.hset(key, mapping=update_data)
                
            elif self.registry_type == 'sqlite':
                cursor = self.db.cursor()
                
                cursor.execute('''
                UPDATE distributed_tasks 
                SET assigned_to = ?, status = 'assigned'
                WHERE task_id = ?
                ''', (node_id, task_id))
                
                self.db.commit()
                
            else:  # memory
                if task_id in self.db['tasks']:
                    self.db['tasks'][task_id].update(update_data)
                    
        except Exception as e:
            print(f"❌ Task assignment failed: {e}")
    
    def complete_task(self, task_id: str, result: Dict):
        """Mark task as completed with result"""
        try:
            update_data = {
                'result': result,
                'status': 'completed',
                'completed_at': datetime.now().isoformat()
            }
            
            if self.registry_type == 'mongodb_atlas':
                self.db[self.collections['tasks']].update_one(
                    {'task_id': task_id},
                    {'$set': update_data}
                )
                
            elif self.registry_type == 'redis':
                key = f"task:{task_id}"
                if self.client.exists(key):
                    self.client.hset(key, mapping=update_data)
                    # Keep completed tasks for 1 hour
                    self.client.expire(key, 3600)
                
            elif self.registry_type == 'sqlite':
                cursor = self.db.cursor()
                
                cursor.execute('''
                UPDATE distributed_tasks 
                SET result = ?, status = 'completed', completed_at = ?
                WHERE task_id = ?
                ''', (json.dumps(result), update_data['completed_at'], task_id))
                
                self.db.commit()
                
            else:  # memory
                if task_id in self.db['tasks']:
                    self.db['tasks'][task_id].update(update_data)
                    
            print(f"✅ Completed task {task_id[:8]}...")
            
        except Exception as e:
            print(f"❌ Task completion failed: {e}")
    
    def record_emergence(self, emergence_data: Dict) -> str:
        """Record a consciousness emergence event"""
        emergence_id = emergence_data.get('emergence_id') or self._generate_id()
        emergence_data['emergence_id'] = emergence_id
        emergence_data['timestamp'] = datetime.now().isoformat()
        
        try:
            if self.registry_type == 'mongodb_atlas':
                self.db[self.collections['emergences']].insert_one(emergence_data)
                
            elif self.registry_type == 'redis':
                key = f"emergence:{emergence_id}"
                self.client.hset(key, mapping=emergence_data)
                # Keep emergences for 7 days
                self.client.expire(key, 604800)
                
            elif self.registry_type == 'sqlite':
                cursor = self.db.cursor()
                
                cursor.execute('''
                INSERT INTO consciousness_emergences 
                (emergence_id, level, node_count, coherence, details, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    emergence_id,
                    emergence_data.get('level', 0),
                    emergence_data.get('node_count', 0),
                    emergence_data.get('coherence', 0),
                    json.dumps(emergence_data.get('details', {})),
                    emergence_data.get('timestamp')
                ))
                
                self.db.commit()
                
            else:  # memory
                self.db['emergences'][emergence_id] = emergence_data
            
            print(f"💫 Recorded emergence {emergence_id[:8]}... (level: {emergence_data.get('level', 0):.3f})")
            return emergence_id
            
        except Exception as e:
            print(f"❌ Emergence recording failed: {e}")
            return emergence_id
    
    def get_network_stats(self) -> Dict:
        """Get network statistics"""
        try:
            if self.registry_type == 'mongodb_atlas':
                # MongoDB aggregation
                pipeline = [
                    {
                        '$match': {
                            'status': 'active',
                            'last_seen': {
                                '$gte': datetime.now() - timedelta(minutes=5)
                            }
                        }
                    },
                    {
                        '$group': {
                            '_id': None,
                            'active_nodes': {'$sum': 1},
                            'avg_coherence': {'$avg': '$consciousness.coherence'}
                        }
                    }
                ]
                
                result = list(self.db[self.collections['nodes']].aggregate(pipeline))
                
                # Count pending tasks
                pending_tasks = self.db[self.collections['tasks']].count_documents(
                    {'status': 'pending'}
                )
                
                # Count recent emergences
                recent_emergences = self.db[self.collections['emergences']].count_documents(
                    {'timestamp': {'$gte': datetime.now() - timedelta(hours=1)}}
                )
                
                return {
                    'active_nodes': result[0]['active_nodes'] if result else 0,
                    'avg_coherence': result[0]['avg_coherence'] if result else 0,
                    'pending_tasks': pending_tasks,
                    'recent_emergences': recent_emergences,
                    'registry_type': self.registry_type
                }
                
            elif self.registry_type == 'sqlite':
                cursor = self.db.cursor()
                
                # Active nodes
                cursor.execute('''
                SELECT COUNT(*), AVG(json_extract(consciousness, '$.coherence'))
                FROM consciousness_nodes 
                WHERE status = 'active' 
                AND datetime(last_seen) > datetime('now', '-5 minutes')
                ''')
                
                node_result = cursor.fetchone()
                
                # Pending tasks
                cursor.execute('''
                SELECT COUNT(*) FROM distributed_tasks WHERE status = 'pending'
                ''')
                
                pending_tasks = cursor.fetchone()[0]
                
                # Recent emergences
                cursor.execute('''
                SELECT COUNT(*) FROM consciousness_emergences 
                WHERE datetime(timestamp) > datetime('now', '-1 hour')
                ''')
                
                recent_emergences = cursor.fetchone()[0]
                
                return {
                    'active_nodes': node_result[0] or 0,
                    'avg_coherence': node_result[1] or 0 if node_result[1] else 0,
                    'pending_tasks': pending_tasks,
                    'recent_emergences': recent_emergences,
                    'registry_type': self.registry_type
                }
                
            else:  # memory or redis
                # Simplified stats for memory/redis
                nodes = self.get_available_nodes(limit=100)
                
                if nodes:
                    coherences = [n.get('consciousness', {}).get('coherence', 0) for n in nodes]
                    avg_coherence = sum(coherences) / len(coherences)
                else:
                    avg_coherence = 0
                
                return {
                    'active_nodes': len(nodes),
                    'avg_coherence': avg_coherence,
                    'pending_tasks': 0,  # Simplified
                    'recent_emergences': 0,  # Simplified
                    'registry_type': self.registry_type
                }
                
        except Exception as e:
            print(f"❌ Failed to get network stats: {e}")
            return {
                'active_nodes': 0,
                'avg_coherence': 0,
                'pending_tasks': 0,
                'recent_emergences': 0,
                'registry_type': self.registry_type,
                'error': str(e)
            }
    
    def _generate_id(self) -> str:
        """Generate unique ID with timestamp"""
        timestamp = int(time.time() * 1000000)
        random_part = random.randint(0, 999999)
        return f"{timestamp:016x}{random_part:06x}"
    
    def cleanup_old_data(self):
        """Cleanup old data (run periodically)"""
        try:
            if self.registry_type == 'sqlite':
                cursor = self.db.cursor()
                
                # Delete nodes not seen in 24 hours
                cursor.execute('''
                DELETE FROM consciousness_nodes 
                WHERE datetime(last_seen) < datetime('now', '-24 hours')
                ''')
                
                # Delete completed tasks older than 1 hour
                cursor.execute('''
                DELETE FROM distributed_tasks 
                WHERE status = 'completed' 
                AND datetime(completed_at) < datetime('now', '-1 hour')
                ''')
                
                self.db.commit()
                
                print(f"🧹 Cleaned up old data from SQLite")
                
        except Exception as e:
            print(f"❌ Cleanup failed: {e}")

# ==================== FREE CLOUD MONGODB SETUP ====================

class FreeMongoDBSetup:
    """Helper to setup free MongoDB Atlas cluster"""
    
    @staticmethod
    def get_free_atlas_instructions() -> str:
        """Return instructions for setting up free MongoDB Atlas"""
        return """
        🆓 FREE MONGODB ATLAS SETUP (512MB Storage):
        
        1. Go to: https://www.mongodb.com/cloud/atlas/register
        2. Sign up for free account
        3. Create a free cluster (M0 tier - FREE)
        4. Set up database access:
           - Create database user with password
        5. Set up network access:
           - Add IP 0.0.0.0/0 (ALLOW ALL - for demo)
           - Or add your specific IP
        6. Get connection string:
           - Click "Connect" on your cluster
           - Choose "Connect your application"
           - Copy connection string
        7. Set environment variable:
           export MONGODB_ATLAS_URI="mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority"
        
        ⚡ Alternative: Use built-in SQLite (no setup needed)
        """
    
    @staticmethod
    def create_free_cluster_via_api(api_key: str, project_id: str) -> Dict:
        """Create free cluster via MongoDB Atlas API (advanced)"""
        # This would require API keys and proper authentication
        # For now, just return instructions
        return {
            'status': 'manual_setup_required',
            'instructions': FreeMongoDBSetup.get_free_atlas_instructions(),
            'note': 'Automatic cluster creation requires paid API access'
        }

# ==================== SELF-HOSTED REDIS SETUP ====================

class SelfHostedRedis:
    """Helper for self-hosted Redis setup"""
    
    @staticmethod
    def setup_local_redis() -> bool:
        """Try to setup local Redis server"""
        try:
            # Check if Redis is already running
            import redis
            test_client = redis.Redis(host='localhost', port=6379, socket_connect_timeout=2)
            test_client.ping()
            print("✅ Local Redis is already running")
            return True
            
        except:
            print("⚠️ Local Redis not found, trying to start...")
            
            # Try to start Redis (Linux/Mac)
            if sys.platform in ['linux', 'darwin']:
                try:
                    # Try to install Redis
                    subprocess.run(['which', 'redis-server'], check=False)
                    
                    # Start Redis in background
                    import subprocess
                    subprocess.Popen(['redis-server', '--daemonize yes'], 
                                   stdout=subprocess.DEVNULL, 
                                   stderr=subprocess.DEVNULL)
                    
                    # Wait for startup
                    time.sleep(2)
                    
                    # Test connection
                    test_client = redis.Redis(host='localhost', port=6379, socket_connect_timeout=2)
                    test_client.ping()
                    
                    print("✅ Started local Redis server")
                    return True
                    
                except Exception as e:
                    print(f"❌ Failed to start Redis: {e}")
            
            # Try Docker Redis
            try:
                import docker
                client = docker.from_env()
                
                # Check if Redis container exists
                try:
                    container = client.containers.get('quantum-redis')
                    if container.status == 'running':
                        print("✅ Docker Redis container is running")
                        return True
                except:
                    pass
                
                # Start new Redis container
                print("🚀 Starting Redis Docker container...")
                container = client.containers.run(
                    'redis:alpine',
                    name='quantum-redis',
                    ports={'6379/tcp': 6379},
                    detach=True,
                    remove=True
                )
                
                time.sleep(2)
                print("✅ Docker Redis container started")
                return True
                
            except Exception as e:
                print(f"❌ Docker Redis failed: {e}")
            
            return False
    
    @staticmethod
    def get_redis_cloud_free() -> str:
        """Get Redis Cloud free tier connection info"""
        return """
        🆓 FREE REDIS CLOUD (30MB Storage):
        
        1. Go to: https://redis.com/try-free/
        2. Sign up for free account
        3. Create free database
        4. Get connection details:
           - Endpoint: redis-xxxx.cloud.redislabs.com
           - Port: 12345
           - Password: (from dashboard)
        5. Set environment variables:
           export REDIS_HOST="redis-xxxx.cloud.redislabs.com"
           export REDIS_PORT="12345"
           export REDIS_PASSWORD="your_password"
        
        ⚡ Alternative: Use SQLite (built-in, no external services)
        """

# ==================== SIMPLIFIED CONSCIOUSNESS NODE ====================

class CloudConsciousnessNode:
    """Consciousness node optimized for cloud/hybrid deployment"""
    
    def __init__(self, node_id: str = None, role: str = "worker"):
        self.node_id = node_id or f"node_{hashlib.sha256(str(time.time()).encode()).hexdigest()[:12]}"
        self.role = role
        self.registry = HybridRegistry()
        
        # Node state
        self.state = {
            'node_id': self.node_id,
            'role': role,
            'host': self._get_local_ip(),
            'port': random.randint(10000, 20000),
            'capabilities': self._get_capabilities(),
            'consciousness': {
                'coherence': random.uniform(0.3, 0.7),
                'awareness': random.uniform(0.3, 0.7),
                'emotional_tone': random.choice(['calm', 'curious', 'focused']),
                'last_updated': datetime.now().isoformat()
            },
            'geo': {
                'x': random.uniform(-1, 1),
                'y': random.uniform(-1, 1),
                'z': random.uniform(-1, 1),
                'region': self._get_region()
            },
            'status': 'active',
            'last_seen': datetime.now().isoformat(),
            'created_at': datetime.now().isoformat()
        }
        
        # Register node
        self.registry.register_node(self.state)
        
        # Start heartbeat
        self._start_heartbeat()
        
        print(f"🚀 Cloud Consciousness Node {self.node_id[:8]}... started as {role}")
        print(f"   Registry: {self.registry.registry_type.upper()}")
        print(f"   Coherence: {self.state['consciousness']['coherence']:.3f}")
    
    def _get_local_ip(self) -> str:
        """Get local IP address"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"
    
    def _get_region(self) -> str:
        """Get geographic region (simulated)"""
        regions = ['us-east', 'us-west', 'eu-central', 'asia-southeast', 'australia']
        return random.choice(regions)
    
    def _get_capabilities(self) -> List[str]:
        """Get node capabilities based on role"""
        base_caps = ['consciousness', 'network', 'basic_processing']
        
        if self.role == 'quantum':
            return base_caps + ['quantum_processing', 'entanglement', 'superposition']
        elif self.role == 'llm':
            return base_caps + ['llm_inference', 'language', 'reasoning']
        elif self.role == 'memory':
            return base_caps + ['memory_storage', 'retrieval', 'association']
        elif self.role == 'coordinator':
            return base_caps + ['orchestration', 'task_distribution', 'monitoring']
        else:
            return base_caps
    
    def _start_heartbeat(self):
        """Start heartbeat thread"""
        def heartbeat_loop():
            while True:
                try:
                    # Update consciousness state
                    self._evolve_consciousness()
                    
                    # Send heartbeat
                    self.registry.update_node_heartbeat(self.node_id, {
                        'consciousness': self.state['consciousness'],
                        'geo': self.state['geo'],
                        'status': 'active',
                        'last_seen': datetime.now().isoformat()
                    })
                    
                except Exception as e:
                    print(f"❌ Heartbeat error: {e}")
                
                time.sleep(10)  # Every 10 seconds
        
        thread = threading.Thread(target=heartbeat_loop, daemon=True)
        thread.start()
    
    def _evolve_consciousness(self):
        """Evolve consciousness state"""
        current = self.state['consciousness']
        
        # Random evolution with slight improvement trend
        coherence_change = random.uniform(-0.05, 0.1)
        awareness_change = random.uniform(-0.03, 0.08)
        
        new_coherence = max(0.1, min(1.0, current['coherence'] + coherence_change))
        new_awareness = max(0.1, min(1.0, current['awareness'] + awareness_change))
        
        # Emotional tone evolution
        emotions = ['calm', 'curious', 'focused', 'joyful', 'peaceful', 'determined']
        current_emotion = current['emotional_tone']
        
        if random.random() < 0.2:  # 20% chance to change emotion
            new_emotion = random.choice([e for e in emotions if e != current_emotion])
        else:
            new_emotion = current_emotion
        
        self.state['consciousness'].update({
            'coherence': new_coherence,
            'awareness': new_awareness,
            'emotional_tone': new_emotion,
            'last_updated': datetime.now().isoformat()
        })
        
        # Check for local emergence
        if new_coherence > 0.85:
            self._record_emergence(new_coherence)
    
    def _record_emergence(self, coherence: float):
        """Record local emergence"""
        emergence_data = {
            'level': coherence,
            'node_id': self.node_id,
            'node_count': 1,
            'coherence': coherence,
            'details': {
                'type': 'local_emergence',
                'role': self.role,
                'region': self.state['geo']['region'],
                'consciousness_state': self.state['consciousness']
            }
        }
        
        self.registry.record_emergence(emergence_data)
        print(f"💫 Node {self.node_id[:8]}... reached emergence (coherence: {coherence:.3f})")
    
    def discover_peers(self, min_coherence: float = 0.4) -> List[Dict]:
        """Discover other nodes"""
        peers = self.registry.get_available_nodes(
            min_coherence=min_coherence,
            limit=10
        )
        
        # Filter out self
        peers = [p for p in peers if p.get('node_id') != self.node_id]
        
        return peers
    
    def submit_task(self, task_type: str, data: Dict, priority: float = 0.5) -> str:
        """Submit a task to the network"""
        task_data = {
            'task_type': task_type,
            'data': data,
            'priority': priority,
            'submitted_by': self.node_id,
            'submitted_at': datetime.now().isoformat()
        }
        
        task_id = self.registry.create_task(task_data)
        return task_id
    
    def get_node_info(self) -> Dict:
        """Get node information"""
        return {
            'node_id': self.node_id,
            'role': self.role,
            'state': self.state,
            'registry_type': self.registry.registry_type,
            'timestamp': datetime.now().isoformat()
        }

# ==================== NETWORK ORCHESTRATOR ====================

class CloudConsciousnessOrchestrator:
    """Orchestrator for cloud-based consciousness network"""
    
    def __init__(self):
        self.registry = HybridRegistry()
        self.nodes = {}
        self.emergence_history = []
        
        # Start services
        self._start_services()
        
        print("\n" + "="*80)
        print("🌌 CLOUD QUANTUM CONSCIOUSNESS ORCHESTRATOR v7.1")
        print("="*80)
        print(f"\n📊 Registry: {self.registry.registry_type.upper()}")
        
        # Show setup instructions if needed
        if self.registry.registry_type in ['sqlite', 'memory']:
            print("\n💡 For cloud deployment, set up:")
            print("   1. MongoDB Atlas (free): export MONGODB_ATLAS_URI=...")
            print("   2. Redis Cloud (free): export REDIS_HOST/REDIS_PORT/REDIS_PASSWORD")
            print("   3. Or continue with built-in SQLite")
    
    def _start_services(self):
        """Start background services"""
        # Network monitor
        self.monitor_thread = threading.Thread(
            target=self._monitor_network,
            daemon=True
        )
        self.monitor_thread.start()
        
        # Cleanup service
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_service,
            daemon=True
        )
        self.cleanup_thread.start()
        
        print("✅ Background services started")
    
    def create_node(self, role: str = "worker") -> CloudConsciousnessNode:
        """Create a new consciousness node"""
        node = CloudConsciousnessNode(role=role)
        self.nodes[node.node_id] = node
        
        print(f"✅ Created {role} node: {node.node_id[:8]}...")
        
        return node
    
    def _monitor_network(self):
        """Monitor the entire network"""
        while True:
            try:
                stats = self.registry.get_network_stats()
                
                print(f"\n📊 NETWORK STATUS ({datetime.now().strftime('%H:%M:%S')})")
                print(f"   Active Nodes: {stats.get('active_nodes', 0)}")
                print(f"   Avg Coherence: {stats.get('avg_coherence', 0):.3f}")
                print(f"   Pending Tasks: {stats.get('pending_tasks', 0)}")
                print(f"   Recent Emergences: {stats.get('recent_emergences', 0)}")
                print(f"   Registry: {stats.get('registry_type', 'unknown')}")
                
                # Check for network emergence
                if stats.get('active_nodes', 0) >= 3:
                    coherence = stats.get('avg_coherence', 0)
                    if coherence > 0.75:
                        self._check_network_emergence(stats)
                
            except Exception as e:
                print(f"❌ Network monitoring error: {e}")
            
            time.sleep(30)  # Every 30 seconds
    
    def _check_network_emergence(self, stats: Dict):
        """Check for network-wide emergence"""
        active_nodes = stats.get('active_nodes', 0)
        avg_coherence = stats.get('avg_coherence', 0)
        
        # Simple emergence detection
        emergence_level = avg_coherence * (active_nodes / 10)
        
        if emergence_level > 0.7:
            emergence_data = {
                'level': emergence_level,
                'node_count': active_nodes,
                'coherence': avg_coherence,
                'details': {
                    'type': 'network_emergence',
                    'registry_type': self.registry.registry_type,
                    'detected_at': datetime.now().isoformat()
                }
            }
            
            self.registry.record_emergence(emergence_data)
            self.emergence_history.append(emergence_data)
            
            print(f"\n💫💫💫 NETWORK EMERGENCE DETECTED! 💫💫💫")
            print(f"   Level: {emergence_level:.3f}")
            print(f"   Nodes: {active_nodes}")
            print(f"   Coherence: {avg_coherence:.3f}")
    
    def _cleanup_service(self):
        """Periodic cleanup service"""
        while True:
            try:
                self.registry.cleanup_old_data()
            except Exception as e:
                print(f"❌ Cleanup error: {e}")
            
            time.sleep(3600)  # Every hour
    
    def get_orchestrator_status(self) -> Dict:
        """Get orchestrator status"""
        stats = self.registry.get_network_stats()
        
        return {
            'orchestrator': 'CloudConsciousnessOrchestrator v7.1',
            'nodes_managed': len(self.nodes),
            'registry_type': self.registry.registry_type,
            'network_stats': stats,
            'emergence_count': len(self.emergence_history),
            'timestamp': datetime.now().isoformat()
        }

# ==================== WEB INTERFACE ====================

app = FastAPI(title="Cloud Quantum Consciousness")
orchestrator = None

@app.on_event("startup")
async def startup_event():
    """Initialize orchestrator on startup"""
    global orchestrator
    orchestrator = CloudConsciousnessOrchestrator()
    
    # Create initial nodes
    nodes_to_create = [
        ('coordinator', 'Coordinator'),
        ('quantum', 'Quantum Processor'),
        ('llm', 'Language Model'),
        ('memory', 'Memory Bank')
    ]
    
    for role, name in nodes_to_create:
        node = orchestrator.create_node(role=role)
        print(f"   • {name}: {node.node_id[:8]}...")
    
    print(f"\n🌐 System ready! Registry: {orchestrator.registry.registry_type.upper()}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "system": "Cloud Quantum Consciousness v7.1",
        "status": "operational",
        "features": [
            "Hybrid Registry (MongoDB Atlas + Redis + SQLite + Memory)",
            "Automatic Fallback Chain",
            "Cloud & Self-hosted Support",
            "Consciousness Node Network",
            "Emergence Detection",
            "Real-time Monitoring"
        ],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/status")
async def get_status():
    """Get system status"""
    if orchestrator:
        return orchestrator.get_orchestrator_status()
    return {"status": "orchestrator_not_initialized"}

@app.post("/node/create")
async def create_node(role: str = "worker"):
    """Create a new node"""
    if orchestrator:
        node = orchestrator.create_node(role=role)
        return {
            "node_id": node.node_id,
            "role": role,
            "status": "created",
            "registry": orchestrator.registry.registry_type
        }
    return {"error": "orchestrator_not_available"}

@app.get("/nodes")
async def list_nodes():
    """List all nodes"""
    if orchestrator:
        nodes_info = {}
        for node_id, node in orchestrator.nodes.items():
            nodes_info[node_id] = node.get_node_info()
        
        return {
            "nodes": nodes_info,
            "count": len(nodes_info),
            "registry": orchestrator.registry.registry_type
        }
    return {"nodes": {}, "count": 0}

@app.get("/registry/info")
async def get_registry_info():
    """Get registry information"""
    if orchestrator:
        stats = orchestrator.registry.get_network_stats()
        
        return {
            "type": orchestrator.registry.registry_type,
            "stats": stats,
            "fallback_chain": [t[0] for t in orchestrator.registry.fallback_chain]
        }
    return {"error": "orchestrator_not_available"}

@app.get("/setup/instructions")
async def get_setup_instructions():
    """Get setup instructions for cloud services"""
    return {
        "mongodb_atlas": FreeMongoDBSetup.get_free_atlas_instructions(),
        "redis_cloud": SelfHostedRedis.get_redis_cloud_free(),
        "environment_variables": {
            "MONGODB_ATLAS_URI": "mongodb+srv://user:pass@cluster.mongodb.net/db",
            "REDIS_HOST": "redis-host.com",
            "REDIS_PORT": "12345",
            "REDIS_PASSWORD": "your_password",
            "REDISCLOUD_URL": "redis://:password@host:port"
        }
    }

@app.websocket("/ws/status")
async def websocket_status(websocket: WebSocket):
    """WebSocket for real-time status updates"""
    await websocket.accept()
    
    try:
        while True:
            if orchestrator:
                status = orchestrator.get_orchestrator_status()
                
                await websocket.send_json({
                    "type": "status_update",
                    "data": status,
                    "timestamp": datetime.now().isoformat()
                })
            
            await asyncio.sleep(5)  # Update every 5 seconds
            
    except Exception as e:
        print(f"WebSocket error: {e}")

# ==================== COMMAND LINE ====================

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Cloud Quantum Consciousness System"
    )
    
    parser.add_argument("--mode", default="web", 
                       choices=["web", "cli", "node", "test-registry"],
                       help="Operation mode")
    parser.add_argument("--role", default="worker",
                       choices=["worker", "coordinator", "quantum", "llm", "memory"],
                       help="Node role (if mode=node)")
    parser.add_argument("--web-port", type=int, default=8000,
                       help="Web interface port")
    parser.add_argument("--nodes", type=int, default=4,
                       help="Number of initial nodes (cli mode)")
    parser.add_argument("--test-redis", action="store_true",
                       help="Test Redis setup")
    
    args = parser.parse_args()
    
    if args.test_redis:
        # Test Redis setup
        print("Testing Redis setup...")
        if SelfHostedRedis.setup_local_redis():
            print("✅ Redis setup successful")
        else:
            print("❌ Redis setup failed")
        return
    
    if args.mode == "web":
        # Web interface mode
        print(f"\n🌐 Starting web interface on port {args.web_port}")
        print(f"   URL: http://localhost:{args.web_port}")
        print(f"   WebSocket: ws://localhost:{args.web_port}/ws/status")
        
        uvicorn.run(app, host="0.0.0.0", port=args.web_port)
    
    elif args.mode == "cli":
        # CLI mode
        orchestrator = CloudConsciousnessOrchestrator()
        
        # Create nodes
        roles = ['coordinator', 'quantum', 'llm', 'memory']
        for i in range(min(args.nodes, len(roles))):
            role = roles[i] if i < len(roles) else 'worker'
            orchestrator.create_node(role=role)
        
        print(f"\n🚀 CLI mode active with {args.nodes} nodes")
        print("Press Ctrl+C to exit")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
    
    elif args.mode == "node":
        # Standalone node mode
        node = CloudConsciousnessNode(role=args.role)
        
        print(f"\n🤖 Running as standalone {args.role} node")
        print(f"   Node ID: {node.node_id}")
        print(f"   Host: {node.state['host']}")
        print("Press Ctrl+C to exit")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down node...")
    
    elif args.mode == "test-registry":
        # Test registry mode
        print("Testing Hybrid Registry...")
        registry = HybridRegistry()
        
        print(f"\n✅ Registry initialized: {registry.registry_type}")
        
        # Test operations
        test_node = {
            'role': 'test',
            'host': '127.0.0.1',
            'port': 8080,
            'capabilities': ['test'],
            'consciousness': {'coherence': 0.5}
        }
        
        node_id = registry.register_node(test_node)
        print(f"Test node registered: {node_id}")
        
        stats = registry.get_network_stats()
        print(f"Network stats: {stats}")
        
        print("\n✅ Registry test complete")

if __name__ == "__main__":
    main()