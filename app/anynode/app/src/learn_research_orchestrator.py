# orchestrator.py
from fastapi import FastAPI
import requests
from typing import List, Optional

app = FastAPI()

SERVICES = {
    "research": "http://localhost:8001",
    "learn": "http://localhost:8000", 
    "invent": "http://localhost:8002"
}

@app.get("/health")
def health_check():
    status = {}
    for service, url in SERVICES.items():
        try:
            response = requests.get(f"{url}/health", timeout=2)
            status[service] = response.json()
        except:
            status[service] = {"status": "down"}
    return status

@app.post("/orchestrate")
def orchestrate_ai(topic: str, seed_facts: List[str], needs: List[str], feedback: Optional[List[str]] = None):
    """Orchestrate the full AI pipeline: Research → Learn → Invent"""
    
    # Step 1: Research the topic
    research_result = requests.post(
        f"{SERVICES['research']}/research",
        json={"topic": topic, "facts": seed_facts}
    ).json()
    
    # Step 2: Learn from the research
    # Convert facts to feature vectors (simplified)
    research_facts = research_result["facts"][-10:]  # Last 10 facts
    features = [[len(fact)] for fact in research_facts]  # Simple feature: length
    
    learn_result = requests.post(
        f"{SERVICES['learn']}/learn",
        json={"features": features}
    ).json()
    
    # Step 3: Invent based on needs and research
    invent_result = requests.post(
        f"{SERVICES['invent']}/invent",
        json={"needs": needs, "feedback": feedback}
    ).json()
    
    return {
        "research": {
            "topic": research_result["topic"],
            "fact_count": research_result["fact_count"]
        },
        "learning": {
            "method": learn_result["method"],
            "loss": learn_result.get("loss")
        },
        "invention": {
            "idea": invent_result["idea"],
            "score": invent_result["score"],
            "valid": invent_result.get("valid", True)
        },
        "pipeline": "complete"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)