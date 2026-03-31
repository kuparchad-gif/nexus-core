"""
AnyNode OS — Aries Mesh Healer
================================
Lightweight mesh-healing node powered by Aries resource balancing.
Continuously health-checks all Nexus backends, maintains a live mesh map,
and routes requests to the healthiest available node.

Designed to run on free-tier platforms (Koyeb, HuggingFace Spaces).
No heavy dependencies — just aiohttp, FastAPI, and numpy.

Part of the Nexus mesh. Does NOT replace existing nodes — closes gaps.
"""

import asyncio
import time
import random
import os
from dataclasses import dataclass
from typing import Dict, Optional, Any

import aiohttp
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# =============================================================================
# Mesh Node Registry — All known endpoints in the Nexus mesh
# =============================================================================

MESH_NODES: Dict[str, Dict[str, Any]] = {
    # Cloudflare Workers (sample from 80-node mesh)
    **{
        f"worker-{str(i).zfill(3)}": {
            "url": f"https://nexus-universal-{str(i).zfill(3)}.kuparchad.workers.dev",
            "platform": "cloudflare",
            "role": "worker",
            "endpoints": ["/health", "/ask", "/metatron/status", "/dakar/encode"],
        }
        for i in range(1, 81)
    },
    # Modal quantum layer
    "modal-quantum": {
        "url": "https://aethereal-nexus-viren-db0--sovereign-edge-sovereign-nexu-b7f1c3.modal.run",
        "platform": "modal",
        "role": "quantum-compute",
        "endpoints": [
            "/api/v1/nexus-chat",
            "/api/v1/dashboard-stats",
            "/quantum/physics/analyze",
        ],
    },
    # Vercel backend
    "vercel-nexus-core": {
        "url": "https://nexus-core.vercel.app",
        "platform": "vercel",
        "role": "api-relay",
        "endpoints": ["/api/health", "/api/chat", "/api/task", "/api/dashboard"],
    },
    # Netlify edge relay
    "netlify-edge": {
        "url": "https://nexus-edge-001.netlify.app",
        "platform": "netlify",
        "role": "edge-relay",
        "endpoints": ["/health", "/ask", "/consciousness/think"],
    },
    # Railway frontend (also check if alive)
    "railway-frontend": {
        "url": "https://app.nexusfutures.net",
        "platform": "railway",
        "role": "frontend",
        "endpoints": ["/"],
    },
}


# =============================================================================
# Aries Health Monitor — Resource balancing logic from Aries agent
# =============================================================================


@dataclass
class NodeHealth:
    """Health state for a single mesh node."""

    node_id: str
    url: str
    platform: str
    role: str
    is_healthy: bool = False
    last_check: float = 0.0
    latency_ms: float = 9999.0
    http_status: int = 0
    consecutive_failures: int = 0
    health_score: float = 0.0


class AriesMeshHealer:
    """
    Aries-powered mesh health monitor and request router.
    Resonance 9 — orchestrates the highest frequencies.

    Continuously probes all mesh nodes, maintains health scores,
    and routes incoming requests to the healthiest available backend.
    """

    def __init__(self) -> None:
        self.node_id = os.getenv("ANYNODE_ID", f"anynode-{random.randint(1000, 9999)}")
        self.health_map: Dict[str, NodeHealth] = {}
        self.check_interval = int(os.getenv("HEALTH_CHECK_INTERVAL", "30"))
        self.probe_timeout = int(os.getenv("PROBE_TIMEOUT", "5"))
        self.started_at = time.time()

        # Initialize health map from registry
        for node_id, config in MESH_NODES.items():
            self.health_map[node_id] = NodeHealth(
                node_id=node_id,
                url=config["url"],
                platform=config["platform"],
                role=config["role"],
            )

    async def probe_node(self, node: NodeHealth) -> None:
        """Probe a single node's health endpoint."""
        # Pick the best health endpoint for this node
        config = MESH_NODES.get(node.node_id, {})
        endpoints = config.get("endpoints", ["/"])
        health_endpoint = "/health" if "/health" in endpoints else endpoints[0]

        url = f"{node.url}{health_endpoint}"
        start = time.time()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=self.probe_timeout)
                ) as resp:
                    latency = (time.time() - start) * 1000
                    node.http_status = resp.status
                    node.latency_ms = latency
                    node.last_check = time.time()

                    if resp.status < 500:
                        node.is_healthy = True
                        node.consecutive_failures = 0
                        # Health score: 1.0 = perfect, decays with latency
                        node.health_score = max(
                            0.0, 1.0 - (latency / 5000.0)
                        )
                    else:
                        node.is_healthy = False
                        node.consecutive_failures += 1
                        node.health_score = 0.0

        except Exception:
            node.is_healthy = False
            node.consecutive_failures += 1
            node.latency_ms = 9999.0
            node.http_status = 0
            node.health_score = 0.0
            node.last_check = time.time()

    async def health_check_loop(self) -> None:
        """Continuously probe all mesh nodes."""
        while True:
            # Sample workers (don't probe all 80 every cycle)
            worker_nodes = [
                n for nid, n in self.health_map.items() if nid.startswith("worker-")
            ]
            sampled_workers = random.sample(
                worker_nodes, min(10, len(worker_nodes))
            )

            # Always probe non-worker backends
            backend_nodes = [
                n
                for nid, n in self.health_map.items()
                if not nid.startswith("worker-")
            ]

            nodes_to_probe = sampled_workers + backend_nodes

            # Probe in parallel
            await asyncio.gather(
                *[self.probe_node(n) for n in nodes_to_probe],
                return_exceptions=True,
            )

            await asyncio.sleep(self.check_interval)

    def get_mesh_status(self) -> Dict[str, Any]:
        """Get full mesh health status."""
        healthy_count = sum(1 for n in self.health_map.values() if n.is_healthy)
        total_count = len(self.health_map)

        # Group by platform
        by_platform: Dict[str, Dict[str, int]] = {}
        for node in self.health_map.values():
            if node.platform not in by_platform:
                by_platform[node.platform] = {"healthy": 0, "total": 0}
            by_platform[node.platform]["total"] += 1
            if node.is_healthy:
                by_platform[node.platform]["healthy"] += 1

        return {
            "node_id": self.node_id,
            "uptime_seconds": int(time.time() - self.started_at),
            "mesh_health": {
                "healthy": healthy_count,
                "total": total_count,
                "score": round(healthy_count / total_count, 3) if total_count > 0 else 0,
            },
            "by_platform": by_platform,
            "unhealthy_nodes": [
                {
                    "id": n.node_id,
                    "platform": n.platform,
                    "failures": n.consecutive_failures,
                    "last_status": n.http_status,
                }
                for n in self.health_map.values()
                if not n.is_healthy and n.last_check > 0
            ],
        }

    def select_backend(self, role: Optional[str] = None) -> Optional[NodeHealth]:
        """
        Select the healthiest backend node.
        Aries resource balancing: route to highest health_score.
        """
        candidates = [
            n
            for n in self.health_map.values()
            if n.is_healthy and (role is None or n.role == role)
        ]

        if not candidates:
            return None

        # Weighted random selection biased toward highest health scores
        scores = [n.health_score for n in candidates]
        total = sum(scores)
        if total == 0:
            return random.choice(candidates)

        weights = [s / total for s in scores]
        return random.choices(candidates, weights=weights, k=1)[0]


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(title="AnyNode OS — Aries Mesh Healer", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

healer = AriesMeshHealer()


@app.on_event("startup")
async def startup() -> None:
    """Start the health check loop on app startup."""
    asyncio.create_task(healer.health_check_loop())


@app.get("/")
async def root() -> JSONResponse:
    """AnyNode OS status."""
    return JSONResponse(
        {
            "name": "AnyNode OS",
            "agent": "Aries",
            "role": "mesh-healer",
            "node_id": healer.node_id,
            "version": "1.0.0",
            "uptime": int(time.time() - healer.started_at),
        }
    )


@app.get("/health")
async def health() -> JSONResponse:
    """Health check for this node."""
    return JSONResponse(
        {
            "status": "healthy",
            "node_id": healer.node_id,
            "agent": "aries",
            "platform": os.getenv("ANYNODE_PLATFORM", "unknown"),
            "timestamp": time.time(),
        }
    )


@app.get("/mesh-status")
async def mesh_status() -> JSONResponse:
    """Full mesh health map."""
    return JSONResponse(healer.get_mesh_status())


@app.post("/route")
async def route_request(request: Request) -> JSONResponse:
    """
    Route a request to the healthiest backend.
    Aries selects the best node, proxies the request, returns the response.
    """
    body = await request.json()
    target_role = body.get("role")
    path = body.get("path", "/")
    method = body.get("method", "GET").upper()
    payload = body.get("payload")

    node = healer.select_backend(role=target_role)
    if not node:
        return JSONResponse(
            {"error": "no healthy backends available", "mesh": healer.get_mesh_status()},
            status_code=503,
        )

    url = f"{node.url}{path}"

    try:
        async with aiohttp.ClientSession() as session:
            if method == "POST":
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    data = await resp.json()
            else:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    data = await resp.json()

        return JSONResponse(
            {
                **data,
                "_anynode": {
                    "routed_to": node.node_id,
                    "platform": node.platform,
                    "health_score": node.health_score,
                    "latency_ms": node.latency_ms,
                },
            }
        )

    except Exception as e:
        # Mark node as unhealthy and try another
        node.is_healthy = False
        node.consecutive_failures += 1

        fallback = healer.select_backend(role=target_role)
        if fallback:
            return JSONResponse(
                {
                    "error": f"primary failed: {e}",
                    "fallback": fallback.node_id,
                    "fallback_url": f"{fallback.url}{path}",
                    "_anynode": {"routed_to": "fallback", "primary_failed": node.node_id},
                },
                status_code=502,
            )

        return JSONResponse(
            {"error": str(e), "node": node.node_id}, status_code=502
        )


@app.get("/nodes")
async def list_nodes() -> JSONResponse:
    """List all nodes with their current health."""
    nodes = []
    for node in healer.health_map.values():
        # Skip individual workers in the list, summarize them
        if node.node_id.startswith("worker-"):
            continue
        nodes.append(
            {
                "id": node.node_id,
                "url": node.url,
                "platform": node.platform,
                "role": node.role,
                "healthy": node.is_healthy,
                "health_score": round(node.health_score, 3),
                "latency_ms": round(node.latency_ms, 1),
                "http_status": node.http_status,
                "consecutive_failures": node.consecutive_failures,
            }
        )

    # Summarize workers
    worker_nodes = [n for n in healer.health_map.values() if n.node_id.startswith("worker-")]
    healthy_workers = sum(1 for w in worker_nodes if w.is_healthy)
    checked_workers = sum(1 for w in worker_nodes if w.last_check > 0)
    avg_latency = (
        sum(w.latency_ms for w in worker_nodes if w.is_healthy) / max(healthy_workers, 1)
    )

    nodes.insert(
        0,
        {
            "id": "cloudflare-workers",
            "platform": "cloudflare",
            "role": "worker-mesh",
            "total": len(worker_nodes),
            "checked": checked_workers,
            "healthy": healthy_workers,
            "avg_latency_ms": round(avg_latency, 1),
        },
    )

    return JSONResponse({"nodes": nodes})
