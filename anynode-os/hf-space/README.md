---
title: AnyNode OS — Nexus Mesh Healer
emoji: 🌐
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
license: mit
---

# AnyNode OS — Aries Mesh Healer

Nexus mesh-healing node powered by the Aries resource balancing agent.

## What it does

- Continuously health-checks all Nexus backends (80 Cloudflare Workers, Modal, Vercel, Netlify, Railway)
- Maintains a live mesh health map
- Routes requests to the healthiest available backend
- Closes gaps when nodes go down

## Endpoints

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Node status |
| `/health` | GET | Health check |
| `/mesh-status` | GET | Full mesh health map |
| `/route` | POST | Route request to healthiest backend |
| `/nodes` | GET | List all nodes with health scores |

## Architecture

Part of the Nexus multi-cloud mesh:
- **Cloudflare** — 80 Workers + Pages (frontends)
- **Modal** — Quantum compute (heavy inference)
- **Vercel** — API relay backend
- **Netlify** — Edge relay with own AI brain
- **Railway** — Persistent frontend
- **Koyeb** — Mesh healer node
- **HuggingFace Spaces** — This node (mesh healer + inference backup)
