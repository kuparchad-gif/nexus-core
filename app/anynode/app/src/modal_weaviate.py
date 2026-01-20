import modal
import os

# Create Modal image with Weaviate dependencies
image = modal.Image.debian_slim().pip_install(
    "weaviate-client", "fastapi", "uvicorn"
)

app = modal.App("cloud-viren-weaviate", image=image)

# Environment variables
WEAVIATE_VERSION = "1.19.6"  # Update to latest stable version as needed
WEAVIATE_PORT = 8080
PERSISTENCE_PATH = "/weaviate-data"

# Create a volume for data persistence
volume = modal.Volume.from_name("weaviate-data", create_if_missing=True)

@app.function(
    volumes={PERSISTENCE_PATH: volume},
    cpu=2.0,
    memory=4096,
)
@modal.asgi_app()
def weaviate_server():
    from fastapi import FastAPI, Request, Response
    import httpx
    import time
    import subprocess
    import signal
    import sys
    
    fast_app = FastAPI()
    weaviate_process = None
    
    @fast_app.on_event("startup")
    async def startup_event():
        nonlocal weaviate_process
        
        # Start Weaviate in the background
        env = os.environ.copy()
        env["AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED"] = "true"
        env["PERSISTENCE_DATA_PATH"] = f"{PERSISTENCE_PATH}/data"
        env["ENABLE_MODULES"] = "text2vec-transformers"
        
        weaviate_process = subprocess.Popen(
            [
                "docker", "run",
                "-e", "AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true",
                "-e", "PERSISTENCE_DATA_PATH=/data",
                "-e", "ENABLE_MODULES=text2vec-transformers",
                "-v", f"{PERSISTENCE_PATH}/data:/data",
                "-p", f"{WEAVIATE_PORT}:{WEAVIATE_PORT}",
                f"semitechnologies/weaviate:{WEAVIATE_VERSION}"
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for Weaviate to start
        print("Starting Weaviate...")
        time.sleep(10)  # Give it time to initialize
        
        # Check if Weaviate is running
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"http://localhost:{WEAVIATE_PORT}/v1/meta")
                if response.status_code == 200:
                    print("Weaviate is running!")
                else:
                    print(f"Weaviate startup issue: {response.status_code}")
        except Exception as e:
            print(f"Error checking Weaviate status: {str(e)}")
    
    @fast_app.on_event("shutdown")
    def shutdown_event():
        if weaviate_process:
            weaviate_process.terminate()
            weaviate_process.wait()
            print("Weaviate stopped")
    
    # Proxy all requests to Weaviate
    @fast_app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"])
    async def proxy(request: Request, path: str):
        url = f"http://localhost:{WEAVIATE_PORT}/{path}"
        
        # Forward the request to Weaviate
        async with httpx.AsyncClient() as client:
            weaviate_response = await client.request(
                method=request.method,
                url=url,
                headers={k: v for k, v in request.headers.items() if k.lower() != "host"},
                content=await request.body(),
                params=request.query_params,
                follow_redirects=True
            )
            
            # Return the response from Weaviate
            return Response(
                content=weaviate_response.content,
                status_code=weaviate_response.status_code,
                headers=dict(weaviate_response.headers)
            )
    
    return fast_app

if __name__ == "__main__":
    modal.runner.deploy_stub(app)