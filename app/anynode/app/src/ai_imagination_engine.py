import torch
import numpy as np
from fastapi import FastAPI
from qdrant_client import QdrantClient
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()
qdrant = QdrantClient(host="localhost", port=6333)

# Need Detection with Creative Spark
def detect_need(input_hints: list):
    base_vector = torch.rand(100).numpy()  # Seed with a creative nudge
    for hint in input_hints:
        tweak = np.random.normal(0, 0.2, 100) * (ord(hint[0]) / 100)  # Imaginative twist
        base_vector += tweak
    points = [{"id": i, "vector": base_vector, "payload": {"need": hint}} for i, hint in enumerate(input_hints)]
    qdrant.upsert(collection_name="nexus_imagine", points=points)
    return base_vector / np.linalg.norm(base_vector)  # Normalize for cohesion

# Ideation with Inventive Flair
def ideate_solution(need_vector):
    solutions = []
    for _ in range(3):
        creative_shift = np.random.normal(0, 0.3, 100) * np.sin(np.arange(100) / 10)  # Wave-like imagination
        idea_vector = need_vector + creative_shift
        score = np.random.uniform(0.7, 1.0) * 0.4 + np.random.uniform(0, 0.3) * 0.4 + np.random.uniform(0, 0.2) * 0.2  # Weighted creativity
        solutions.append({"vector": idea_vector, "score": score, "idea": f" Invention_{_}"})
    best_idea = max(solutions, key=lambda x: x["score"])
    qdrant.upsert(collection_name="nexus_imagine", points=[{"id": 100+i, "vector": s["vector"], "payload": {"idea": s["idea"], "score": s["score"]}} for i, s in enumerate(solutions)])
    return best_idea

# Improvement with Feedback
def enhance_idea(idea, feedback: list):
    feedback_influence = torch.tensor([0.2 if "good" in f.lower() else -0.2 for f in feedback]).numpy()
    refined_vector = idea["vector"] + feedback_influence * 0.6  # Amplify positive feedback
    qdrant.upsert(collection_name="nexus_imagine", points=[{"id": 103, "vector": refined_vector, "payload": {"idea": idea["idea"] + "_Enhanced", "score": idea["score"] + 0.15}}])
    return {"vector": refined_vector, "idea": idea["idea"] + "_Enhanced"}

# Validation with Soulful Metrics
def validate_invention(enhanced_idea, thresholds: dict = {"impact": 0.9, "feasibility": 0.8}):
    sim_scores = cosine_similarity([enhanced_idea["vector"]], [torch.rand(100).numpy() for _ in range(5)])
    overall_score = np.mean(sim_scores) * 100
    is_valid = all(overall_score > v * 100 for v in thresholds.values())
    return {"valid": is_valid, "score": overall_score}

# API Endpoints
@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/imagine")
def invent_with_imagination(needs: list, feedback: list = None):
    need_vector = detect_need(needs)
    initial_idea = ideate_solution(need_vector)
    if feedback:
        enhanced = enhance_idea(initial_idea, feedback)
        result = validate_invention(enhanced)
        qdrant.upsert(collection_name="nexus_imagine", points=[{"id": 104, "vector": enhanced["vector"], "payload": {"idea": enhanced["idea"], "valid": result["valid"], "score": result["score"]}}])
        return {"idea": enhanced["idea"], "valid": result["valid"], "score": result["score"]}
    return {"idea": initial_idea["idea"], "score": initial_idea["score"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)