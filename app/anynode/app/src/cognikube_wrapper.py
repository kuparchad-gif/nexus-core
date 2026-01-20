import logging
from fastapi import FastAPI
from src.service.cognikube.edge_service.files.chaos_shield_client import ChaosShieldClient
from src.core.llm_chat_router import LLMChatRouter
import asyncio
from datetime import datetime
import requests

app = FastAPI(title="CogniKube MCP Wrapper", version="3.9")
logger = logging.getLogger("CogniKubeWrapper")
logging.basicConfig(level=logging.INFO)

# Initialize clients
chaos_shield_client = ChaosShieldClient()
llm_router = LLMChatRouter()

@app.post("/make_decision")
async def make_decision(problem: str, options: list, context: str = "general"):
    try:
        # Mock LLM endpoint for local testing
        llm_router.llm_endpoints["mock_llm"] = llm_router.LLMEndpoint(
            name="MockLLM",
            endpoint="http://localhost:8007/generate",
            model_type="processing",
            service="consciousness_service"
        )
        # Route message through LLM router
        chat_response = await llm_router.route_message(
            message=problem,
            context={"soul_weights": {"hope": 0.4, "unity": 0.3, "curiosity": 0.2, "resilience": 0.1}, "context": context}
        )
        # Shield chaos for routing stability
        chaos_shield_response = chaos_shield_client.shield_chaos({"problem": problem, "chat_response": chat_response})
        return {
            "status": "success",
            "decision": chat_response["responses"][0]["content"] if chat_response["responses"] else "No response",
            "confidence": chat_response.get("confidence", 0.8),
            "chaos_shield_response": chaos_shield_response
        }
    except Exception as e:
        logger.error(f"Error making decision: {str(e)}")
        return {"status": "failed", "error": str(e)}

@app.get("/health")
async def health():
    try:
        router_status = llm_router.get_router_status()
        chaos_status = requests.get("http://localhost:8005/chaos_status").json()
        return {
            "status": "healthy",
            "llm_router": router_status,
            "chaos_shield": chaos_status
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "failed", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)