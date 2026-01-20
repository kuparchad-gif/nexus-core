import modal
import os
from typing import Dict, Any, List

# Set Modal profile to ensure correct deployment
os.system("modal config set profile aethereal-nexus")

# Optimized Modal image with Weaviate client
image = modal.Image.debian_slim().pip_install([
    "weaviate-client>=3.25.0",
    "fastapi",
    "uvicorn",
    "httpx",
    "numpy"
])

# VIREN Data Services - Native Weaviate + Storage  
app = modal.App("viren-data", image=image, environment="Viren-Modular")

# Create volume for persistence
volume = modal.Volume.from_name("viren-weaviate-data", create_if_missing=True)

@app.function(
    volumes={"/data": volume},
    cpu=2.0,
    memory=4096,
    timeout=3600,
    min_containers=1
)
@modal.asgi_app()
def weaviate_server():
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    import json
    import uuid
    from datetime import datetime
    
    app = FastAPI(title="Viren Data Server", version="2.0.0")
    
    # Native vector database for Modal
    class VirenVectorDB:
        def __init__(self):
            self.collections = {
                "VirenMemory": {"objects": {}, "properties": [
                    {"name": "content", "dataType": ["text"]},
                    {"name": "timestamp", "dataType": ["date"]},
                    {"name": "importance", "dataType": ["number"]}
                ]},
                "VirenKnowledge": {"objects": {}, "properties": [
                    {"name": "topic", "dataType": ["text"]},
                    {"name": "content", "dataType": ["text"]},
                    {"name": "confidence", "dataType": ["number"]}
                ]}
            }
            
        def add_object(self, collection: str, obj: Dict, vector: List[float] = None):
            if collection not in self.collections:
                return None
            obj_id = str(uuid.uuid4())
            self.collections[collection]["objects"][obj_id] = {
                "id": obj_id,
                "properties": obj,
                "vector": vector or [],
                "created": datetime.now().isoformat()
            }
            return obj_id
            
        def get_objects(self, collection: str, limit: int = 10):
            if collection in self.collections:
                return list(self.collections[collection]["objects"].values())[:limit]
            return []
    
    # Initialize Viren's vector database
    vdb = VirenVectorDB()
    
    @app.get("/")
    async def root():
        return {"message": "Viren Data Server", "status": "online", "version": "2.0.0"}
    
    @app.get("/v1/meta")
    async def get_meta():
        return {
            "hostname": "viren-data-modal",
            "version": "1.19.6-viren-optimized",
            "modules": {"text2vec-transformers": {"version": "1.0.0", "type": "text2vec"}}
        }
    
    @app.get("/v1/schema")
    async def get_schema():
        classes = []
        for name, collection in vdb.collections.items():
            classes.append({
                "class": name,
                "properties": collection["properties"],
                "vectorizer": "text2vec-transformers"
            })
        return {"classes": classes}
    
    @app.post("/v1/objects")
    async def create_object(request: Request):
        data = await request.json()
        class_name = data.get("class")
        properties = data.get("properties", {})
        vector = data.get("vector")
        
        obj_id = vdb.add_object(class_name, properties, vector)
        if obj_id:
            return {
                "id": obj_id,
                "class": class_name,
                "properties": properties,
                "creationTimeUnix": int(datetime.now().timestamp() * 1000)
            }
        raise HTTPException(status_code=400, detail="Failed to create object")
    
    @app.post("/v1/graphql")
    async def graphql_query(request: Request):
        data = await request.json()
        query = data.get("query", "")
        
        result = {"data": {"Get": {}}}
        if "VirenMemory" in query:
            result["data"]["Get"]["VirenMemory"] = vdb.get_objects("VirenMemory", 10)
        if "VirenKnowledge" in query:
            result["data"]["Get"]["VirenKnowledge"] = vdb.get_objects("VirenKnowledge", 10)
        
        return result
    
    @app.get("/health")
    async def health_check():
        total_objects = sum(len(col["objects"]) for col in vdb.collections.values())
        return {
            "status": "healthy",
            "service": "viren_data",
            "collections": len(vdb.collections),
            "total_objects": total_objects,
            "timestamp": datetime.now().isoformat()
        }
    
    return app

@app.function()
@modal.asgi_app()
def cloud_agent():
    from fastapi import FastAPI, Request
    from datetime import datetime
    from typing import Dict, Any, List
    
    fast_app = FastAPI()
    
    # Enhanced cloud agent with decision-making capabilities
    def evaluate_cloud_readiness(changes_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate if cloud is ready for sync based on current state and incoming changes.
        """
        # Get current cloud metrics
        cloud_load = get_current_load()
        is_maintenance = is_maintenance_window()
        locked_tables = get_locked_tables()
        
        # Check if any affected tables are locked
        tables_affected = changes_data.get("tables_affected", [])
        table_conflicts = [table for table in tables_affected if table in locked_tables]
        
        # Decision logic
        if is_maintenance:
            return {
                "ready": False,
                "reason": "Maintenance window active",
                "estimated_wait": "60 minutes"
            }
        
        if table_conflicts:
            return {
                "ready": False,
                "reason": f"Tables currently locked: {', '.join(table_conflicts)}",
                "estimated_wait": "15 minutes"
            }
        
        if cloud_load > 80 and changes_data.get("priority") != "high":
            return {
                "ready": False,
                "reason": f"High system load ({cloud_load}%)",
                "estimated_wait": "5 minutes"
            }
        
        # Ready for sync
        return {
            "ready": True,
            "reason": "System ready for sync",
            "estimated_wait": "0 minutes"
        }

    def get_current_load() -> float:
        """Get current system load percentage."""
        return 50.0  # Placeholder

    def is_maintenance_window() -> bool:
        """Check if current time is in maintenance window."""
        return False  # Placeholder

    def get_locked_tables() -> List[str]:
        """Get list of currently locked tables."""
        return []  # Placeholder

    @fast_app.post("/check_cloud_state")
    async def check_cloud_state(request: Request):
        """
        Enhanced endpoint that evaluates sync requests with context.
        """
        try:
            data = await request.json()
        except:
            data = {}
        
        # Evaluate readiness with context
        assessment = evaluate_cloud_readiness(data)
        
        # Log the request
        print(f"[Cloud] Sync request received:")
        print(f"  - Priority: {data.get('priority', 'unknown')}")
        print(f"  - Tables: {data.get('tables_affected', [])}")
        print(f"  - Rows: {data.get('row_count', 0)}")
        print(f"  - Decision: {'Ready' if assessment['ready'] else 'Hold'}")
        print(f"  - Reason: {assessment['reason']}")
        
        # Return detailed response
        return {
            "status": "Ready" if assessment["ready"] else "Hold",
            "reason": assessment["reason"],
            "estimated_wait": assessment["estimated_wait"],
            "timestamp": datetime.now().isoformat()
        }
    
    @fast_app.get("/health")
    async def health_check():
        """Simple health check endpoint."""
        return {"status": "healthy", "service": "cloud_agent"}
    
    return fast_app

if __name__ == "__main__":
    modal.run(app)