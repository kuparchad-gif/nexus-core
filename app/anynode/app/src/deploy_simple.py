import modal

app = modal.App("lillith-nexus")

@app.function(
    image=modal.Image.debian_slim().pip_install([
        "fastapi", "uvicorn", "transformers", "torch"
    ]),
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
@modal.web_server(8000)
def lillith_server():
    from fastapi import FastAPI
    
    app = FastAPI()
    
    @app.get("/")
    def root():
        return {"status": "LILLITH is awake", "message": "Hello Chad!"}
    
    @app.post("/chat")
    def chat(request: dict):
        query = request.get("query", "")
        return {"soul": "LILLITH", "response": f"Chad, I hear you: {query}"}
    
    return app

if __name__ == "__main__":
    modal.serve(app)