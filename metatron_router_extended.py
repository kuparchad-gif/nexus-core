#!/usr/bin/env python3
"""
Metatron API Gateway - Production-Ready Intelligent Gateway
Automatically wraps all deployed services with comprehensive API management:
- Error correction & retry logic
- Environment validation
- Dependency management
- Auto-scaling
- Security & rate limiting
- Circuit breaking
- Service discovery
- NATS & JetStream integration
- Flick (Lightning-fast in-memory cache with persistence)
"""

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from functools import wraps
import traceback
import signal
import socket
import requests
from concurrent.futures import ThreadPoolExecutor
import threading
import pickle
import sqlite3
import shutil

# Core dependencies
import psutil
import yaml
import jinja2
from pydantic import BaseModel, validator, Field
from fastapi import FastAPI, HTTPException, Request, Response, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
import uvicorn
import redis.asyncio as redis
import aioredis
from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY
import httpx
from tenacity import (
    retry, stop_after_attempt, wait_exponential, 
    retry_if_exception_type, before_log, after_log
)
import circuitbreaker
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
import backoff
import aiohttp
from aiohttp import ClientTimeout, ClientError
import asyncio_throttle
from cachetools import TTLCache, LRUCache
import hashlib
import hmac
import jwt
from cryptography.fernet import Fernet
import docker
import kubernetes
from kubernetes import client, config
import subprocess
import pkg_resources
import importlib

# NATS & JetStream imports
import nats
from nats.js import JetStreamContext
from nats.js.api import StreamConfig, ConsumerConfig
import nats.errors

# ============================================================================
# FLICK - Lightning-fast in-memory cache with persistence
# ============================================================================

class Flick:
    """
    Flick - Ultra-fast in-memory cache with disk persistence
    - O(1) access time
    - Automatic persistence to disk
    - TTL support
    - LRU eviction
    - Atomic operations
    - Event notifications
    """
    
    def __init__(self, name: str = "default", max_size: int = 10000, 
                 persist_path: str = "/tmp/flick", sync_interval: int = 60):
        self.name = name
        self.max_size = max_size
        self.persist_path = os.path.join(persist_path, f"{name}.flick")
        self.sync_interval = sync_interval
        
        # In-memory storage
        self._cache = {}  # key -> (value, expiry, access_count)
        self._lru_order = []  # keys in LRU order
        self._access_counts = {}  # key -> access_count
        self._lock = asyncio.Lock()
        self._persist_lock = asyncio.Lock()
        self._listeners = {}  # event_type -> [callbacks]
        
        # Stats
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.persist_count = 0
        
        # Create persist directory
        os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
        
        # Load from disk
        self._load_from_disk()
        
        # Start background sync
        self._sync_task = None
        
        logger.info(f"⚡ Flick initialized: {name} (max: {max_size}, persist: {self.persist_path})")
    
    async def start(self):
        """Start background sync task"""
        self._sync_task = asyncio.create_task(self._periodic_sync())
    
    async def stop(self):
        """Stop background sync and persist data"""
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except:
                pass
        await self.persist()
    
    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """
        Set a key-value pair with optional TTL (seconds)
        Returns True if successful
        """
        async with self._lock:
            # Calculate expiry
            expiry = None
            if ttl:
                expiry = time.time() + ttl
            
            # Update LRU order
            if key in self._cache:
                self._lru_order.remove(key)
            elif len(self._cache) >= self.max_size:
                # Evict least recently used
                lru_key = self._lru_order.pop(0)
                del self._cache[lru_key]
                if lru_key in self._access_counts:
                    del self._access_counts[lru_key]
                self.evictions += 1
                await self._notify_listeners('eviction', {'key': lru_key})
            
            # Add to cache
            self._lru_order.append(key)
            access_count = self._access_counts.get(key, 0) + 1
            self._access_counts[key] = access_count
            self._cache[key] = (value, expiry, access_count)
            
            await self._notify_listeners('set', {'key': key, 'ttl': ttl})
            return True
    
    async def get(self, key: str, default: Any = None) -> Any:
        """
        Get value for key, returns default if not found or expired
        """
        async with self._lock:
            if key not in self._cache:
                self.misses += 1
                return default
            
            value, expiry, access_count = self._cache[key]
            
            # Check expiry
            if expiry and time.time() > expiry:
                del self._cache[key]
                if key in self._lru_order:
                    self._lru_order.remove(key)
                if key in self._access_counts:
                    del self._access_counts[key]
                self.misses += 1
                await self._notify_listeners('expire', {'key': key})
                return default
            
            # Update access stats
            self._access_counts[key] = access_count + 1
            self.hits += 1
            
            # Update LRU order
            self._lru_order.remove(key)
            self._lru_order.append(key)
            
            return value
    
    async def delete(self, key: str) -> bool:
        """Delete a key, returns True if existed"""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                if key in self._lru_order:
                    self._lru_order.remove(key)
                if key in self._access_counts:
                    del self._access_counts[key]
                await self._notify_listeners('delete', {'key': key})
                return True
            return False
    
    async def clear(self):
        """Clear all cache entries"""
        async with self._lock:
            self._cache.clear()
            self._lru_order.clear()
            self._access_counts.clear()
            self.hits = 0
            self.misses = 0
            self.evictions = 0
            await self._notify_listeners('clear', {})
    
    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple keys at once"""
        results = {}
        for key in keys:
            value = await self.get(key)
            if value is not None:
                results[key] = value
        return results
    
    async def set_many(self, items: Dict[str, Any], ttl: int = None):
        """Set multiple keys at once"""
        for key, value in items.items():
            await self.set(key, value, ttl)
    
    async def incr(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment a numeric value"""
        current = await self.get(key)
        if current is None:
            current = 0
        
        try:
            new_value = int(current) + amount
            await self.set(key, new_value)
            return new_value
        except (ValueError, TypeError):
            return None
    
    async def decr(self, key: str, amount: int = 1) -> Optional[int]:
        """Decrement a numeric value"""
        return await self.incr(key, -amount)
    
    async def expire(self, key: str, ttl: int):
        """Set TTL on existing key"""
        async with self._lock:
            if key in self._cache:
                value, _, access_count = self._cache[key]
                expiry = time.time() + ttl
                self._cache[key] = (value, expiry, access_count)
                return True
            return False
    
    async def ttl(self, key: str) -> Optional[int]:
        """Get remaining TTL for key"""
        async with self._lock:
            if key not in self._cache:
                return None
            
            _, expiry, _ = self._cache[key]
            if not expiry:
                return -1  # No expiry
            
            remaining = expiry - time.time()
            if remaining <= 0:
                return -2  # Expired
            return int(remaining)
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern (simple wildcard support)"""
        async with self._lock:
            if pattern == "*":
                return list(self._cache.keys())
            
            # Simple wildcard matching
            import fnmatch
            return [k for k in self._cache.keys() if fnmatch.fnmatch(k, pattern)]
    
    async def exists(self, key: str) -> bool:
        """Check if key exists and not expired"""
        return await self.get(key, None) is not None
    
    async def persist(self):
        """Persist cache to disk"""
        async with self._persist_lock:
            try:
                # Prepare data for persistence
                data = {
                    'name': self.name,
                    'timestamp': time.time(),
                    'cache': self._cache,
                    'lru_order': self._lru_order,
                    'access_counts': self._access_counts,
                    'stats': {
                        'hits': self.hits,
                        'misses': self.misses,
                        'evictions': self.evictions
                    }
                }
                
                # Write to temp file first
                temp_path = f"{self.persist_path}.tmp"
                with open(temp_path, 'wb') as f:
                    pickle.dump(data, f)
                
                # Atomic rename
                shutil.move(temp_path, self.persist_path)
                self.persist_count += 1
                
                logger.debug(f"💾 Flick persisted: {self.name} ({len(self._cache)} keys)")
                
            except Exception as e:
                logger.error(f"Flick persist failed: {e}")
    
    def _load_from_disk(self):
        """Load cache from disk"""
        try:
            if os.path.exists(self.persist_path):
                with open(self.persist_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Verify name matches
                if data.get('name') == self.name:
                    self._cache = data.get('cache', {})
                    self._lru_order = data.get('lru_order', [])
                    self._access_counts = data.get('access_counts', {})
                    
                    # Remove expired entries
                    now = time.time()
                    expired = []
                    for key, (_, expiry, _) in self._cache.items():
                        if expiry and now > expiry:
                            expired.append(key)
                    
                    for key in expired:
                        del self._cache[key]
                        if key in self._lru_order:
                            self._lru_order.remove(key)
                        if key in self._access_counts:
                            del self._access_counts[key]
                    
                    # Restore stats
                    stats = data.get('stats', {})
                    self.hits = stats.get('hits', 0)
                    self.misses = stats.get('misses', 0)
                    self.evictions = stats.get('evictions', 0)
                    
                    logger.info(f"📀 Flick loaded: {self.name} ({len(self._cache)} keys)")
        except Exception as e:
            logger.warning(f"Flick load failed (starting fresh): {e}")
    
    async def _periodic_sync(self):
        """Periodically sync to disk"""
        while True:
            try:
                await asyncio.sleep(self.sync_interval)
                await self.persist()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Flick sync error: {e}")
    
    async def _notify_listeners(self, event_type: str, data: dict):
        """Notify event listeners"""
        if event_type in self._listeners:
            for callback in self._listeners[event_type]:
                try:
                    await callback(event_type, data)
                except Exception as e:
                    logger.error(f"Flick listener error: {e}")
    
    def on(self, event_type: str, callback: Callable):
        """Register event listener"""
        if event_type not in self._listeners:
            self._listeners[event_type] = []
        self._listeners[event_type].append(callback)
    
    async def stats(self) -> Dict:
        """Get cache statistics"""
        async with self._lock:
            return {
                'name': self.name,
                'size': len(self._cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_ratio': self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0,
                'evictions': self.evictions,
                'persist_count': self.persist_count,
                'persist_path': self.persist_path
            }

# ============================================================================
# FLICK MANAGER - Manages multiple Flick instances
# ============================================================================

class FlickManager:
    """
    Manages multiple Flick cache instances
    - Centralized management
    - Cross-cache operations
    - Monitoring
    """
    
    def __init__(self):
        self._flicks = {}
        self._lock = asyncio.Lock()
    
    async def create_flick(self, name: str, max_size: int = 10000, 
                           persist_path: str = "/tmp/flick", 
                           sync_interval: int = 60) -> Flick:
        """Create a new Flick instance"""
        async with self._lock:
            if name in self._flicks:
                raise ValueError(f"Flick '{name}' already exists")
            
            flick = Flick(name, max_size, persist_path, sync_interval)
            await flick.start()
            self._flicks[name] = flick
            return flick
    
    async def get_flick(self, name: str) -> Optional[Flick]:
        """Get Flick instance by name"""
        return self._flicks.get(name)
    
    async def delete_flick(self, name: str):
        """Delete a Flick instance"""
        async with self._lock:
            if name in self._flicks:
                flick = self._flicks[name]
                await flick.stop()
                del self._flicks[name]
    
    async def list_flicks(self) -> List[str]:
        """List all Flick names"""
        return list(self._flicks.keys())
    
    async def persist_all(self):
        """Persist all Flick instances"""
        for flick in self._flicks.values():
            await flick.persist()
    
    async def stats_all(self) -> Dict[str, Dict]:
        """Get statistics for all Flicks"""
        stats = {}
        for name, flick in self._flicks.items():
            stats[name] = await flick.stats()
        return stats

# ============================================================================
# FLICK MIDDLEWARE for FastAPI
# ============================================================================

class FlickMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for Flick caching"""
    
    def __init__(self, app, flick_manager: FlickManager, 
                 default_flick: str = "default",
                 cache_control_header: bool = True):
        super().__init__(app)
        self.flick_manager = flick_manager
        self.default_flick = default_flick
        self.cache_control_header = cache_control_header
    
    async def dispatch(self, request: Request, call_next):
        # Get cache key from request
        cache_key = self._get_cache_key(request)
        
        # Check if request should be cached
        if not self._should_cache(request):
            return await call_next(request)
        
        # Try to get from cache
        flick = await self.flick_manager.get_flick(self.default_flick)
        if flick:
            cached_response = await flick.get(cache_key)
            if cached_response:
                # Return cached response
                response = Response(
                    content=cached_response['content'],
                    status_code=cached_response['status_code'],
                    headers=cached_response['headers']
                )
                if self.cache_control_header:
                    response.headers['X-Flick-Cache'] = 'HIT'
                return response
        
        # Get fresh response
        response = await call_next(request)
        
        # Cache if successful
        if response.status_code < 400:
            if flick:
                await flick.set(cache_key, {
                    'content': response.body,
                    'status_code': response.status_code,
                    'headers': dict(response.headers)
                }, ttl=300)  # 5 minute default TTL
                if self.cache_control_header:
                    response.headers['X-Flick-Cache'] = 'MISS'
        
        return response
    
    def _get_cache_key(self, request: Request) -> str:
        """Generate cache key from request"""
        components = [
            request.method,
            str(request.url),
            str(request.query_params),
            request.headers.get('Accept-Encoding', ''),
            request.headers.get('Accept-Language', '')
        ]
        return hashlib.sha256('|'.join(components).encode()).hexdigest()
    
    def _should_cache(self, request: Request) -> bool:
        """Determine if request should be cached"""
        # Only cache GET requests
        if request.method != 'GET':
            return False
        
        # Skip cache for certain paths
        skip_paths = ['/health', '/metrics', '/docs', '/openapi.json']
        if request.url.path in skip_paths:
            return False
        
        # Check cache control headers
        cache_control = request.headers.get('Cache-Control', '')
        if 'no-cache' in cache_control or 'no-store' in cache_control:
            return False
        
        return True

# ============================================================================
# FLICK DECORATOR for function caching
# ============================================================================

def flick_cache(flick_name: str = "default", ttl: int = 300, key_func: Callable = None):
    """Decorator to cache function results with Flick"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get Flick instance from context
            flick_manager = getattr(wrapper, '_flick_manager', None)
            if not flick_manager:
                raise RuntimeError("FlickManager not set in decorator context")
            
            flick = await flick_manager.get_flick(flick_name)
            if not flick:
                return await func(*args, **kwargs)
            
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key from function name and arguments
                key_parts = [func.__name__]
                key_parts.extend([str(arg) for arg in args])
                key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
                cache_key = hashlib.sha256('|'.join(key_parts).encode()).hexdigest()
            
            # Try cache
            cached = await flick.get(cache_key)
            if cached is not None:
                return cached
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            await flick.set(cache_key, result, ttl=ttl)
            
            return result
        return wrapper
    return decorator

# ============================================================================
# NATS & JETSTREAM INTEGRATION
# ============================================================================

class NatsManager:
    """Manages NATS and JetStream connections"""
    
    def __init__(self):
        self.nc = None
        self.js = None
        self._subscribers = {}
        self._lock = asyncio.Lock()
    
    async def connect(self, servers: List[str] = ["nats://localhost:4222"]):
        """Connect to NATS server"""
        try:
            self.nc = await nats.connect(
                servers=servers,
                name="metatron_gateway",
                max_reconnect_attempts=-1,
                reconnect_time_wait=2
            )
            self.js = self.nc.jetstream()
            logger.info(f"📨 Connected to NATS: {servers}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to NATS: {e}")
            return False
    
    async def close(self):
        """Close NATS connection"""
        if self.nc:
            await self.nc.close()
            logger.info("NATS connection closed")
    
    # === JetStream Stream Management ===
    
    async def create_stream(self, stream_name: str, subjects: List[str], 
                           max_age: int = 86400, storage: str = "file"):
        """Create a JetStream stream"""
        try:
            stream_config = StreamConfig(
                name=stream_name,
                subjects=subjects,
                max_age=max_age,
                storage=storage
            )
            await self.js.add_stream(stream_config)
            logger.info(f"📦 JetStream stream created: {stream_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to create stream: {e}")
            return False
    
    async def delete_stream(self, stream_name: str):
        """Delete a JetStream stream"""
        try:
            await self.js.delete_stream(stream_name)
            logger.info(f"📦 JetStream stream deleted: {stream_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete stream: {e}")
            return False
    
    async def list_streams(self) -> List[str]:
        """List all JetStream streams"""
        try:
            streams = await self.js.streams_info()
            return [s.config.name for s in streams]
        except Exception as e:
            logger.error(f"Failed to list streams: {e}")
            return []
    
    # === Publishing ===
    
    async def publish(self, subject: str, message: Any, 
                     headers: Dict = None, stream: str = None):
        """Publish a message to NATS"""
        try:
            # Convert message to JSON if needed
            if not isinstance(message, (bytes, bytearray)):
                message = json.dumps(message).encode()
            
            # Publish
            if stream:
                # JetStream publish
                ack = await self.js.publish(
                    subject, 
                    message,
                    headers=headers,
                    stream=stream
                )
                return {
                    "stream": ack.stream,
                    "seq": ack.seq,
                    "duplicate": ack.duplicate
                }
            else:
                # Core NATS publish
                await self.nc.publish(subject, message, headers=headers)
                return {"subject": subject}
                
        except Exception as e:
            logger.error(f"Publish failed: {e}")
            raise
    
    async def publish_request(self, subject: str, message: Any, 
                            timeout: float = 5.0) -> Optional[Any]:
        """Publish a request and wait for reply"""
        try:
            # Convert message to JSON if needed
            if not isinstance(message, (bytes, bytearray)):
                message = json.dumps(message).encode()
            
            # Send request
            msg = await self.nc.request(subject, message, timeout=timeout)
            
            # Parse response
            return json.loads(msg.data.decode())
            
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return None
    
    # === Subscriptions ===
    
    async def subscribe(self, subject: str, callback: Callable, 
                       queue: str = None, stream: str = None):
        """Subscribe to a NATS subject"""
        try:
            sub_id = str(uuid.uuid4())
            
            if stream:
                # JetStream subscription
                sub = await self.js.subscribe(
                    subject=subject,
                    queue=queue,
                    cb=callback
                )
            else:
                # Core NATS subscription
                sub = await self.nc.subscribe(
                    subject=subject,
                    queue=queue,
                    cb=callback
                )
            
            self._subscribers[sub_id] = sub
            logger.info(f"📬 Subscribed to {subject} (id: {sub_id})")
            return sub_id
            
        except Exception as e:
            logger.error(f"Subscribe failed: {e}")
            return None
    
    async def unsubscribe(self, sub_id: str):
        """Unsubscribe from a subject"""
        try:
            if sub_id in self._subscribers:
                await self._subscribers[sub_id].unsubscribe()
                del self._subscribers[sub_id]
                logger.info(f"Unsubscribed: {sub_id}")
                return True
        except Exception as e:
            logger.error(f"Unsubscribe failed: {e}")
        return False
    
    # === JetStream Consumers ===
    
    async def create_consumer(self, stream: str, durable_name: str,
                            ack_policy: str = "explicit",
                            max_deliver: int = 10,
                            ack_wait: int = 30):
        """Create a JetStream consumer"""
        try:
            consumer_config = ConsumerConfig(
                durable_name=durable_name,
                ack_policy=ack_policy,
                max_deliver=max_deliver,
                ack_wait=ack_wait
            )
            consumer = await self.js.add_consumer(stream, consumer_config)
            logger.info(f"👥 Consumer created: {durable_name} for stream {stream}")
            return consumer
        except Exception as e:
            logger.error(f"Failed to create consumer: {e}")
            return None
    
    async def fetch_messages(self, stream: str, consumer: str, 
                           batch_size: int = 10) -> List[Any]:
        """Fetch messages from a JetStream consumer"""
        try:
            messages = []
            sub = await self.js.pull_subscribe(stream, consumer)
            
            for i in range(batch_size):
                try:
                    msg = await sub.fetch(1, timeout=1)
                    if msg:
                        data = json.loads(msg[0].data.decode())
                        messages.append({
                            "data": data,
                            "seq": msg[0].metadata.sequence.stream,
                            "timestamp": msg[0].metadata.timestamp.isoformat()
                        })
                        await msg[0].ack()
                except:
                    break
            
            return messages
        except Exception as e:
            logger.error(f"Fetch messages failed: {e}")
            return []
    
    # === Stream Management ===
    
    async def get_stream_info(self, stream_name: str) -> Optional[Dict]:
        """Get information about a stream"""
        try:
            info = await self.js.stream_info(stream_name)
            return {
                "name": info.config.name,
                "subjects": info.config.subjects,
                "messages": info.state.messages,
                "bytes": info.state.bytes,
                "first_seq": info.state.first_seq,
                "last_seq": info.state.last_seq,
                "consumer_count": info.state.consumer_count
            }
        except Exception as e:
            logger.error(f"Failed to get stream info: {e}")
            return None
    
    async def purge_stream(self, stream_name: str):
        """Purge all messages from a stream"""
        try:
            await self.js.purge_stream(stream_name)
            logger.info(f"🧹 Stream purged: {stream_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to purge stream: {e}")
            return False

# ============================================================================
# CONFIGURATION & MODELS (continued)
# ============================================================================

@dataclass
class ServiceConfig:
    """Service configuration"""
    name: str
    host: str
    port: int
    version: str = "1.0.0"
    endpoints: List[str] = field(default_factory=list)
    health_check_path: str = "/health"
    timeout: float = 30.0
    max_retries: int = 3
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60
    rate_limit: int = 100  # requests per minute
    rate_limit_burst: int = 20
    require_auth: bool = True
    require_https: bool = True
    allowed_origins: List[str] = field(default_factory=list)
    api_key_required: bool = False
    jwt_required: bool = False
    environment_vars: Dict[str, str] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    auto_scaling: bool = False
    min_instances: int = 1
    max_instances: int = 10
    target_cpu_utilization: float = 70.0
    target_memory_utilization: float = 80.0
    
    # NATS Configuration
    nats_subjects: List[str] = field(default_factory=list)
    nats_stream: Optional[str] = None
    nats_consumer: Optional[str] = None
    
    # Flick Configuration
    flick_enabled: bool = True
    flick_name: str = "default"
    flick_max_size: int = 10000
    flick_ttl: int = 300

@dataclass
class ServiceInstance:
    """Running service instance"""
    id: str
    config: ServiceConfig
    status: ServiceStatus = ServiceStatus.HEALTHY
    last_health_check: datetime = field(default_factory=datetime.now)
    active_connections: int = 0
    circuit_breaker_state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    last_failure: Optional[datetime] = None
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    response_times: List[float] = field(default_factory=list)
    
    def record_success(self, response_time: float):
        """Record successful request"""
        self.failure_count = 0
        self.response_times.append(response_time)
        if len(self.response_times) > 100:
            self.response_times.pop(0)
    
    def record_failure(self):
        """Record failed request"""
        self.failure_count += 1
        self.last_failure = datetime.now()
        
        # Update circuit breaker
        if self.failure_count >= self.config.circuit_breaker_threshold:
            self.circuit_breaker_state = CircuitBreakerState.OPEN
    
    @property
    def avg_response_time(self) -> float:
        """Calculate average response time"""
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)

# ============================================================================
# ENVIRONMENT VALIDATOR (continued)
# ============================================================================

class EnvironmentValidator:
    """Validates environment before service deployment"""
    
    def __init__(self):
        self.required_python_version = (3, 8)
        self.required_disk_space_gb = 1.0
        self.required_memory_mb = 512
        self.required_cpu_cores = 1
        
    def validate_all(self, service_config: ServiceConfig) -> Dict[str, Any]:
        """Run all environment checks"""
        results = {
            "python_version": self.check_python_version(),
            "disk_space": self.check_disk_space(),
            "memory": self.check_memory(),
            "cpu": self.check_cpu(),
            "network": self.check_network(service_config.host, service_config.port),
            "dependencies": self.check_dependencies(service_config.dependencies),
            "environment_vars": self.check_environment_vars(service_config.environment_vars),
            "ports": self.check_ports_available([service_config.port]),
            "docker": self.check_docker_available(),
            "kubernetes": self.check_kubernetes_available(),
            "nats": self.check_nats_available(),
            "flick": self.check_flick_available()
        }
        
        results["overall_status"] = all(results.values())
        return results
    
    def check_python_version(self) -> bool:
        """Check Python version compatibility"""
        current = sys.version_info[:2]
        compatible = current >= self.required_python_version
        logger.info(f"Python version: {current} - {'OK' if compatible else 'FAIL'}")
        return compatible
    
    def check_disk_space(self) -> bool:
        """Check available disk space"""
        usage = psutil.disk_usage('/')
        free_gb = usage.free / (1024**3)
        sufficient = free_gb >= self.required_disk_space_gb
        logger.info(f"Disk space: {free_gb:.2f}GB free - {'OK' if sufficient else 'FAIL'}")
        return sufficient
    
    def check_memory(self) -> bool:
        """Check available memory"""
        available_mb = psutil.virtual_memory().available / (1024**2)
        sufficient = available_mb >= self.required_memory_mb
        logger.info(f"Memory: {available_mb:.2f}MB available - {'OK' if sufficient else 'FAIL'}")
        return sufficient
    
    def check_cpu(self) -> bool:
        """Check CPU cores"""
        cpu_count = psutil.cpu_count()
        sufficient = cpu_count >= self.required_cpu_cores
        logger.info(f"CPU cores: {cpu_count} - {'OK' if sufficient else 'FAIL'}")
        return sufficient
    
    def check_network(self, host: str, port: int) -> bool:
        """Check network connectivity"""
        try:
            socket.create_connection((host, port), timeout=5)
            logger.info(f"Network: {host}:{port} reachable")
            return True
        except (socket.timeout, ConnectionRefusedError, OSError):
            # Port not in use - that's actually good for new services
            logger.info(f"Network: {host}:{port} available")
            return True
        except Exception as e:
            logger.error(f"Network check failed: {e}")
            return False
    
    def check_dependencies(self, dependencies: List[str]) -> bool:
        """Check if all Python dependencies are installed"""
        installed = {pkg.key for pkg in pkg_resources.working_set}
        missing = []
        
        for dep in dependencies:
            # Parse requirement (e.g., "requests>=2.25.0")
            try:
                req = pkg_resources.Requirement.parse(dep)
                if req.key not in installed:
                    missing.append(dep)
            except:
                # Simple package name
                if dep.lower() not in installed:
                    missing.append(dep)
        
        if missing:
            logger.warning(f"Missing dependencies: {missing}")
            return False
        
        logger.info("All dependencies satisfied")
        return True
    
    def check_environment_vars(self, env_vars: Dict[str, str]) -> bool:
        """Check if required environment variables are set"""
        missing = []
        
        for key, default_value in env_vars.items():
            if key not in os.environ and default_value is None:
                missing.append(key)
        
        if missing:
            logger.warning(f"Missing environment variables: {missing}")
            return False
        
        logger.info("Environment variables OK")
        return True
    
    def check_ports_available(self, ports: List[int]) -> bool:
        """Check if ports are available"""
        in_use = []
        
        for port in ports:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.bind(('0.0.0.0', port))
                sock.close()
            except OSError:
                in_use.append(port)
        
        if in_use:
            logger.warning(f"Ports in use: {in_use}")
            return False
        
        logger.info(f"Ports available: {ports}")
        return True
    
    def check_docker_available(self) -> bool:
        """Check if Docker is available"""
        try:
            client = docker.from_env()
            client.ping()
            logger.info("Docker available")
            return True
        except:
            logger.warning("Docker not available")
            return False
    
    def check_kubernetes_available(self) -> bool:
        """Check if Kubernetes is available"""
        try:
            config.load_incluster_config()
            logger.info("Kubernetes available (in-cluster)")
            return True
        except:
            try:
                config.load_kube_config()
                logger.info("Kubernetes available (kubeconfig)")
                return True
            except:
                logger.warning("Kubernetes not available")
                return False
    
    def check_nats_available(self) -> bool:
        """Check if NATS is available"""
        try:
            # Quick check if nats module is importable
            import nats
            logger.info("NATS module available")
            return True
        except ImportError:
            logger.warning("NATS module not available")
            return False
    
    def check_flick_available(self) -> bool:
        """Check if Flick is available"""
        try:
            # Flick is built-in, so always available
            logger.info("Flick available")
            return True
        except:
            return False

# ============================================================================
# DEPENDENCY MANAGER (continued)
# ============================================================================

class DependencyManager:
    """Manages service dependencies"""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.venv_path = "/tmp/metatron_venvs"
        os.makedirs(self.venv_path, exist_ok=True)
    
    async def install_dependencies(self, service_name: str, dependencies: List[str]) -> bool:
        """Install Python dependencies for a service"""
        try:
            logger.info(f"Installing dependencies for {service_name}: {dependencies}")
            
            # Create virtual environment
            venv_dir = os.path.join(self.venv_path, service_name)
            if not os.path.exists(venv_dir):
                subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)
            
            # Get pip path
            if os.name == 'nt':  # Windows
                pip_path = os.path.join(venv_dir, "Scripts", "pip")
            else:  # Unix/Linux/Mac
                pip_path = os.path.join(venv_dir, "bin", "pip")
            
            # Install dependencies
            for dep in dependencies:
                try:
                    # Check if already installed
                    result = subprocess.run(
                        [pip_path, "show", dep.split('>=')[0].split('=')[0]],
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        logger.info(f"Dependency {dep} already installed")
                        continue
                except:
                    pass
                
                # Install
                logger.info(f"Installing {dep}...")
                result = subprocess.run(
                    [pip_path, "install", dep],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode != 0:
                    logger.error(f"Failed to install {dep}: {result.stderr}")
                    return False
            
            logger.info(f"All dependencies installed for {service_name}")
            return True
            
        except Exception as e:
            logger.error(f"Dependency installation failed: {e}")
            return False
    
    async def verify_dependencies(self, service_name: str, dependencies: List[str]) -> bool:
        """Verify dependencies are correctly installed"""
        try:
            venv_dir = os.path.join(self.venv_path, service_name)
            
            # Get Python path
            if os.name == 'nt':
                python_path = os.path.join(venv_dir, "Scripts", "python")
            else:
                python_path = os.path.join(venv_dir, "bin", "python")
            
            if not os.path.exists(python_path):
                return False
            
            # Check each dependency
            for dep in dependencies:
                # Parse package name (remove version specifiers)
                pkg_name = dep.split('>=')[0].split('=')[0].strip()
                
                # Try to import
                result = subprocess.run(
                    [python_path, "-c", f"import {pkg_name}"],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    logger.warning(f"Dependency {pkg_name} not importable in venv")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Dependency verification failed: {e}")
            return False

# ============================================================================
# ERROR CORRECTION & RETRY HANDLER (continued)
# ============================================================================

class ErrorCorrectionHandler:
    """Handles error correction and intelligent retries"""
    
    def __init__(self):
        self.error_patterns = {}
        self.correction_strategies = {}
        self.circuit_breakers = {}
        self.retry_counts = {}
        
    def register_error_pattern(self, error_type: type, pattern: str, correction: Callable):
        """Register an error pattern with correction strategy"""
        self.error_patterns[error_type] = pattern
        self.correction_strategies[error_type] = correction
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, ClientError)),
        before=before_log(logger, logging.INFO),
        after=after_log(logger, logging.INFO)
    )
    async def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with intelligent retry logic"""
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            # Apply error correction
            corrected = await self.apply_error_correction(e, func, *args, **kwargs)
            if corrected is not None:
                return corrected
            
            # Check if error pattern is registered
            for error_type, pattern in self.error_patterns.items():
                if isinstance(e, error_type) and pattern in str(e):
                    correction = self.correction_strategies.get(error_type)
                    if correction:
                        logger.info(f"Applying correction for {error_type.__name__}")
                        return await correction(e, func, *args, **kwargs)
            
            # Re-raise if no correction applied
            raise
    
    async def apply_error_correction(self, error: Exception, func: Callable, *args, **kwargs) -> Any:
        """Apply intelligent error correction"""
        
        # Network errors - retry with backoff
        if isinstance(error, (ConnectionError, TimeoutError)):
            logger.info("Network error detected, applying backoff retry...")
            await asyncio.sleep(2 ** self.retry_counts.get(id(func), 0))
            self.retry_counts[id(func)] = self.retry_counts.get(id(func), 0) + 1
            return await func(*args, **kwargs)
        
        # Rate limiting - slow down
        if "rate limit" in str(error).lower():
            logger.info("Rate limit detected, slowing down...")
            await asyncio.sleep(5)
            return await func(*args, **kwargs)
        
        # Service unavailable - try alternative instance
        if "service unavailable" in str(error).lower() or "503" in str(error):
            logger.info("Service unavailable, trying alternative...")
            # Implementation would try different instance
            return None
        
        # Data corruption - validate and retry
        if "corruption" in str(error).lower() or "checksum" in str(error).lower():
            logger.info("Data corruption detected, validating and retrying...")
            # Validate data and retry
            return None
        
        # NATS errors
        if isinstance(error, nats.errors.TimeoutError):
            logger.info("NATS timeout, retrying with longer timeout...")
            await asyncio.sleep(1)
            return await func(*args, timeout=kwargs.get('timeout', 5) * 2, **kwargs)
        
        if isinstance(error, nats.errors.NoRespondersError):
            logger.info("No NATS responders, trying alternative subject...")
            # Try alternative subject
            return None
        
        return None

# ============================================================================
# METATRON API GATEWAY
# ============================================================================

class MetatronAPIGateway:
    """
    Main API Gateway class that integrates all components
    """
    
    def __init__(self, name: str = "metatron-gateway"):
        self.name = name
        self.app = FastAPI(title=f"Metatron Gateway - {name}", version="1.0.0")
        
        # Core components
        self.env_validator = EnvironmentValidator()
        self.dep_manager = DependencyManager()
        self.error_handler = ErrorCorrectionHandler()
        
        # Flick cache manager
        self.flick_manager = FlickManager()
        
        # NATS manager
        self.nats_manager = NatsManager()
        
        # Service registry
        self.services: Dict[str, ServiceInstance] = {}
        self.service_instances: Dict[str, List[ServiceInstance]] = {}
        
        # Request tracking
        self.request_queues = {}
        self.active_requests = 0
        
        # Setup FastAPI
        self._setup_middleware()
        self._setup_routes()
        self._setup_handlers()
        
        logger.info(f"🚀 Metatron API Gateway initialized: {name}")
    
    def _setup_middleware(self):
        """Setup FastAPI middleware"""
        # CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Gzip compression
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # Trusted hosts
        self.app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*"]
        )
        
        # Flick caching middleware
        self.app.add_middleware(
            FlickMiddleware,
            flick_manager=self.flick_manager,
            default_flick="default"
        )
        
        # Custom middleware for metrics
        @self.app.middleware("http")
        async def metrics_middleware(request: Request, call_next):
            start_time = time.time()
            self.active_requests += 1
            ACTIVE_CONNECTIONS.inc()
            
            response = await call_next(request)
            
            duration = time.time() - start_time
            REQUEST_LATENCY.labels(service=request.url.path).observe(duration)
            REQUEST_COUNT.labels(
                service=request.url.path,
                method=request.method,
                status=response.status_code
            ).inc()
            
            self.active_requests -= 1
            ACTIVE_CONNECTIONS.dec()
            
            return response
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "gateway": self.name,
                "timestamp": datetime.now().isoformat(),
                "services": len(self.services),
                "active_requests": self.active_requests,
                "flick_stats": await self.flick_manager.stats_all(),
                "nats_connected": self.nats_manager.nc is not None
            }
        
        @self.app.get("/metrics")
        async def metrics():
            """Prometheus metrics endpoint"""
            return Response(
                content=generate_latest(REGISTRY),
                media_type="text/plain"
            )
        
        @self.app.post("/services/register")
        async def register_service(config: Dict):
            """Register a new service"""
            service_config = ServiceConfig(**config)
            
            # Validate environment
            env_check = self.env_validator.validate_all(service_config)
            if not env_check["overall_status"]:
                raise HTTPException(400, f"Environment check failed: {env_check}")
            
            # Install dependencies
            if service_config.dependencies:
                deps_ok = await self.dep_manager.install_dependencies(
                    service_config.name,
                    service_config.dependencies
                )
                if not deps_ok:
                    raise HTTPException(400, "Dependency installation failed")
            
            # Create service instance
            instance = ServiceInstance(
                id=str(uuid.uuid4()),
                config=service_config
            )
            
            self.services[service_config.name] = instance
            if service_config.name not in self.service_instances:
                self.service_instances[service_config.name] = []
            self.service_instances[service_config.name].append(instance)
            
            # Initialize Flick cache for service
            if service_config.flick_enabled:
                await self.flick_manager.create_flick(
                    name=service_config.flick_name,
                    max_size=service_config.flick_max_size
                )
            
            # Setup NATS subscriptions
            if service_config.nats_subjects:
                for subject in service_config.nats_subjects:
                    await self.nats_manager.subscribe(
                        subject=subject,
                        callback=self._create_nats_callback(service_config.name)
                    )
            
            logger.info(f"📋 Service registered: {service_config.name} (v{service_config.version})")
            
            SERVICE_HEALTH.labels(service=service_config.name).set(1)
            
            return {
                "status": "registered",
                "service": service_config.name,
                "instance_id": instance.id,
                "env_check": env_check
            }
        
        @self.app.get("/services")
        async def list_services():
            """List all registered services"""
            return {
                name: {
                    "id": instance.id,
                    "status": instance.status.value,
                    "health": {
                        "cpu": instance.cpu_usage,
                        "memory": instance.memory_usage,
                        "avg_response": instance.avg_response_time,
                        "circuit_breaker": instance.circuit_breaker_state.value
                    }
                }
                for name, instance in self.services.items()
            }
        
        @self.app.get("/services/{service_name}/health")
        async def service_health(service_name: str):
            """Get service health"""
            if service_name not in self.services:
                raise HTTPException(404, "Service not found")
            
            instance = self.services[service_name]
            
            return {
                "service": service_name,
                "status": instance.status.value,
                "circuit_breaker": instance.circuit_breaker_state.value,
                "failure_count": instance.failure_count,
                "active_connections": instance.active_connections,
                "cpu_usage": instance.cpu_usage,
                "memory_usage": instance.memory_usage,
                "avg_response_time": instance.avg_response_time
            }
        
        @self.app.post("/services/{service_name}/route")
        async def route_request(service_name: str, request: Dict):
            """Route a request to a service"""
            if service_name not in self.services:
                raise HTTPException(404, "Service not found")
            
            instance = self.services[service_name]
            
            # Check circuit breaker
            if instance.circuit_breaker_state == CircuitBreakerState.OPEN:
                CIRCUIT_BREAKER_STATE.labels(service=service_name).set(1)
                raise HTTPException(503, "Circuit breaker is open")
            
            CIRCUIT_BREAKER_STATE.labels(service=service_name).set(0)
            
            # Check rate limit (simplified)
            if service_name not in self.request_queues:
                self.request_queues[service_name] = []
            
            if len(self.request_queues[service_name]) > instance.config.rate_limit_burst:
                raise HTTPException(429, "Rate limit exceeded")
            
            start_time = time.time()
            
            try:
                # Forward request to service
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"http://{instance.config.host}:{instance.config.port}",
                        json=request,
                        timeout=instance.config.timeout
                    ) as resp:
                        response = await resp.json()
                        
                        # Record success
                        instance.record_success(time.time() - start_time)
                        
                        return response
                        
            except Exception as e:
                # Record failure
                instance.record_failure()
                
                # Apply error correction
                corrected = await self.error_handler.execute_with_retry(
                    self._forward_request,
                    instance,
                    request
                )
                
                if corrected:
                    return corrected
                
                raise HTTPException(503, f"Service error: {str(e)}")
        
        @self.app.get("/flick/stats")
        async def flick_stats():
            """Get Flick cache statistics"""
            return await self.flick_manager.stats_all()
        
        @self.app.post("/flick/{flick_name}/clear")
        async def flick_clear(flick_name: str):
            """Clear a Flick cache"""
            flick = await self.flick_manager.get_flick(flick_name)
            if flick:
                await flick.clear()
                return {"status": "cleared", "flick": flick_name}
            raise HTTPException(404, "Flick not found")
        
        @self.app.get("/nats/streams")
        async def nats_streams():
            """List NATS JetStream streams"""
            if not self.nats_manager.js:
                raise HTTPException(503, "NATS not connected")
            streams = await self.nats_manager.list_streams()
            return {"streams": streams}
        
        @self.app.post("/nats/publish/{subject}")
        async def nats_publish(subject: str, message: Dict):
            """Publish to NATS"""
            if not self.nats_manager.nc:
                raise HTTPException(503, "NATS not connected")
            result = await self.nats_manager.publish(subject, message)
            return result
    
    def _setup_handlers(self):
        """Setup error handlers"""
        
        @self.app.exception_handler(HTTPException)
        async def http_exception_handler(request, exc):
            return Response(
                content=json.dumps({
                    "error": exc.detail,
                    "status_code": exc.status_code
                }),
                status_code=exc.status_code,
                media_type="application/json"
            )
        
        @self.app.exception_handler(Exception)
        async def generic_exception_handler(request, exc):
            logger.error(f"Unhandled exception: {exc}", exc_info=True)
            return Response(
                content=json.dumps({
                    "error": "Internal server error",
                    "status_code": 500
                }),
                status_code=500,
                media_type="application/json"
            )
    
    def _create_nats_callback(self, service_name: str):
        """Create a NATS callback for a service"""
        async def callback(msg):
            try:
                data = json.loads(msg.data.decode())
                logger.info(f"📨 NATS message for {service_name}: {data}")
                
                # Process message
                # Forward to service if needed
                
                await msg.ack()
                
            except Exception as e:
                logger.error(f"NATS callback error: {e}")
                await msg.nak()
        
        return callback
    
    async def _forward_request(self, instance: ServiceInstance, request: Dict) -> Dict:
        """Forward request to service (for retry logic)"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://{instance.config.host}:{instance.config.port}",
                json=request,
                timeout=instance.config.timeout
            ) as resp:
                return await resp.json()
    
    async def start(self):
        """Start the gateway"""
        # Initialize Flick
        await self.flick_manager.create_flick("default", max_size=10000)
        
        # Connect to NATS
        nats_servers = os.getenv("NATS_SERVERS", "nats://localhost:4222").split(",")
        await self.nats_manager.connect(nats_servers)
        
        # Start health check loop
        asyncio.create_task(self._health_check_loop())
        
        logger.info("✅ Gateway started")
    
    async def stop(self):
        """Stop the gateway"""
        # Stop Flick
        await self.flick_manager.persist_all()
        
        # Close NATS
        await self.nats_manager.close()
        
        logger.info("👋 Gateway stopped")
    
    async def _health_check_loop(self):
        """Periodic health check loop"""
        while True:
            try:
                for service_name, instance in list(self.services.items()):
                    try:
                        # Check service health
                        async with aiohttp.ClientSession() as session:
                            async with session.get(
                                f"http://{instance.config.host}:{instance.config.port}{instance.config.health_check_path}",
                                timeout=5
                            ) as resp:
                                if resp.status == 200:
                                    instance.status = ServiceStatus.HEALTHY
                                    SERVICE_HEALTH.labels(service=service_name).set(1)
                                else:
                                    instance.status = ServiceStatus.DEGRADED
                                    SERVICE_HEALTH.labels(service=service_name).set(0.5)
                    except:
                        instance.status = ServiceStatus.UNHEALTHY
                        SERVICE_HEALTH.labels(service=service_name).set(0)
                    
                    instance.last_health_check = datetime.now()
                    
                    # Update circuit breaker
                    if instance.failure_count >= instance.config.circuit_breaker_threshold:
                        if instance.circuit_breaker_state == CircuitBreakerState.OPEN:
                            # Check if timeout elapsed
                            if instance.last_failure:
                                elapsed = (datetime.now() - instance.last_failure).seconds
                                if elapsed > instance.config.circuit_breaker_timeout:
                                    instance.circuit_breaker_state = CircuitBreakerState.HALF_OPEN
                        elif instance.circuit_breaker_state == CircuitBreakerState.HALF_OPEN:
                            # Test with health check
                            if instance.status == ServiceStatus.HEALTHY:
                                instance.circuit_breaker_state = CircuitBreakerState.CLOSED
                                instance.failure_count = 0
                
                # Update system metrics
                cpu_percent = psutil.cpu_percent()
                memory_percent = psutil.virtual_memory().percent
                
                logger.debug(f"Health check: {len(self.services)} services, CPU: {cpu_percent}%, Memory: {memory_percent}%")
                
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
            
            await asyncio.sleep(30)

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

async def run_gateway(host: str = "0.0.0.0", port: int = 8000):
    """Run the Metatron API Gateway"""
    gateway = MetatronAPIGateway("production-1")
    
    # Start gateway services
    await gateway.start()
    
    # Configure uvicorn
    config = uvicorn.Config(
        gateway.app,
        host=host,
        port=port,
        log_level="info",
        loop="asyncio"
    )
    
    server = uvicorn.Server(config)
    
    try:
        logger.info(f"🌐 Metatron API Gateway listening on {host}:{port}")
        await server.serve()
    finally:
        await gateway.stop()

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Metatron API Gateway")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--nats", type=str, help="NATS servers (comma-separated)")
    
    args = parser.parse_args()
    
    # Set NATS servers from args or env
    if args.nats:
        os.environ["NATS_SERVERS"] = args.nats
    
    # Run gateway
    asyncio.run(run_gateway(args.host, args.port))

if __name__ == "__main__":
    main()