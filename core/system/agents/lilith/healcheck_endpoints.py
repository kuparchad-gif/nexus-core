# Add health check endpoints
@app.function(image=lilith_image, cpu=1, memory=512)
@modal.web_endpoint(method="GET")
def health_check():
    """Comprehensive health check for Lilith's systems"""
    try:
        # Check Qdrant connection
        qdrant_health = requests.get(f"{qdrant_url}/health").status_code == 200
        
        # Check Gabriel network
        gabriel_health = asyncio.run(check_gabriel_connection())
        
        # Check system resources
        system_health = psutil.cpu_percent() < 90 and psutil.virtual_memory().percent < 95
        
        status = "healthy" if all([qdrant_health, gabriel_health, system_health]) else "degraded"
        
        return {
            "status": status,
            "components": {
                "qdrant": "healthy" if qdrant_health else "unhealthy",
                "gabriel_network": "healthy" if gabriel_health else "unhealthy", 
                "system_resources": "healthy" if system_health else "degraded"
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }