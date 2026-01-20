@app.function(image=..., ...)
@modal.web_server(8000)
def api():
    web = FastAPI()

    @web.get("/health")
    async def health():
        return {"status": "ok", "time": datetime.now().isoformat()}

    # ... rest of your endpoints ...
    return web