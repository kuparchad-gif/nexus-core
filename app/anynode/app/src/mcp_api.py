#!/usr/bin/env python3
"""
MCP API for Cloud Viren
Provides REST API for the Vector MCP
"""

import os
import sys
import json
import time
import logging
import uvicorn
from typing import Dict, List, Any, Optional, Union

from fastapi import FastAPI, HTTPException, Depends, Query, Body, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MCPApi")

# Import Vector MCP
try:
    from vector_mcp import VectorMCP
except ImportError:
    logger.error("Vector MCP module not found. Make sure vector_mcp.py is in the same directory.")
    sys.exit(1)

# Create FastAPI app
app = FastAPI(
    title="Cloud Viren MCP API",
    description="API for Cloud Viren Master Control Program",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create Vector MCP instance
mcp = VectorMCP()

# Pydantic models for API
class TextItem(BaseModel):
    text: str
    id: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None

class VectorItem(BaseModel):
    vector: List[float]
    id: str
    payload: Optional[Dict[str, Any]] = None

class BatchItem(BaseModel):
    items: List[Dict[str, Any]]

class NodeItem(BaseModel):
    url: str
    api_key: Optional[str] = None

class SearchQuery(BaseModel):
    query: str
    limit: int = 10
    filter: Optional[Dict[str, Any]] = None

class VectorQuery(BaseModel):
    vector: List[float]
    limit: int = 10
    filter: Optional[Dict[str, Any]] = None

class FilterQuery(BaseModel):
    filter: Dict[str, Any]

# Startup event
@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    logger.info("Starting MCP API")
    mcp.start()

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    logger.info("Shutting down MCP API")
    mcp.stop()

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "timestamp": time.time()
    }

# Status endpoint
@app.get("/status")
async def get_status():
    """Get MCP status"""
    return mcp.get_status()

# Collection endpoints
@app.get("/collections")
async def get_collections():
    """Get all collections"""
    return mcp.get_collection_info()

@app.get("/collections/{collection_name}")
async def get_collection(collection_name: str):
    """Get collection info"""
    try:
        return mcp.get_collection_info(collection_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

# Point endpoints
@app.post("/collections/{collection_name}/points/text")
async def add_text(collection_name: str, item: TextItem):
    """Add text to collection"""
    try:
        # Generate ID if not provided
        if not item.id:
            import uuid
            item.id = str(uuid.uuid4())
        
        success = mcp.add_text(collection_name, item.id, item.text, item.payload)
        
        if success:
            return {"id": item.id, "status": "success"}
        else:
            raise HTTPException(status_code=500, detail="Failed to add text")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/collections/{collection_name}/points/vector")
async def add_vector(collection_name: str, item: VectorItem):
    """Add vector to collection"""
    try:
        success = mcp.add_point(collection_name, item.id, item.vector, item.payload)
        
        if success:
            return {"id": item.id, "status": "success"}
        else:
            raise HTTPException(status_code=500, detail="Failed to add vector")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/collections/{collection_name}/points/batch")
async def add_batch(collection_name: str, batch: BatchItem):
    """Add batch of items to collection"""
    try:
        success = mcp.add_batch(collection_name, batch.items)
        
        if success:
            return {"count": len(batch.items), "status": "success"}
        else:
            raise HTTPException(status_code=500, detail="Failed to add batch")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collections/{collection_name}/points/{point_id}")
async def get_point(collection_name: str, point_id: str):
    """Get point from collection"""
    try:
        point = mcp.get_point(collection_name, point_id)
        
        if point:
            return point
        else:
            raise HTTPException(status_code=404, detail=f"Point {point_id} not found")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/collections/{collection_name}/points/{point_id}")
async def delete_point(collection_name: str, point_id: str):
    """Delete point from collection"""
    try:
        success = mcp.delete_point(collection_name, point_id)
        
        if success:
            return {"id": point_id, "status": "success"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete point")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/collections/{collection_name}/points/delete_by_filter")
async def delete_by_filter(collection_name: str, query: FilterQuery):
    """Delete points by filter"""
    try:
        deleted_count = mcp.delete_by_filter(collection_name, query.filter)
        
        return {"deleted_count": deleted_count, "status": "success"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Search endpoints
@app.post("/collections/{collection_name}/search/text")
async def search_text(collection_name: str, query: SearchQuery):
    """Search by text"""
    try:
        results = mcp.search_text(collection_name, query.query, query.limit, query.filter)
        
        return {"results": results, "count": len(results)}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/collections/{collection_name}/search/vector")
async def search_vector(collection_name: str, query: VectorQuery):
    """Search by vector"""
    try:
        results = mcp.search(collection_name, query.vector, query.limit, query.filter)
        
        return {"results": results, "count": len(results)}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Embedding endpoints
@app.post("/embed")
async def embed_text(item: TextItem):
    """Embed text"""
    try:
        vector = mcp.embed_text(item.text)
        
        return {"vector": vector, "dimension": len(vector)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed/batch")
async def embed_batch(items: List[TextItem]):
    """Embed batch of texts"""
    try:
        texts = [item.text for item in items]
        vectors = mcp.embed_batch(texts)
        
        return {
            "vectors": vectors,
            "count": len(vectors),
            "dimension": len(vectors[0]) if vectors else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Node endpoints
@app.get("/nodes")
async def get_nodes():
    """Get all nodes"""
    return mcp.get_node_status()

@app.post("/nodes")
async def add_node(node: NodeItem):
    """Add replication node"""
    try:
        success = mcp.add_node(node.url, node.api_key)
        
        if success:
            return {"url": node.url, "status": "success"}
        else:
            raise HTTPException(status_code=500, detail="Failed to add node")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/nodes")
async def remove_node(node: NodeItem):
    """Remove replication node"""
    try:
        success = mcp.remove_node(node.url)
        
        if success:
            return {"url": node.url, "status": "success"}
        else:
            raise HTTPException(status_code=404, detail="Node not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the API server
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MCP API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port)