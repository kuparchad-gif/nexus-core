# ai_invention_engine.py
import numpy as np
from fastapi import FastAPI
from .core import qdrant, get_embedding, get_embeddings
from typing import List, Optional

app = FastAPI()

# Need Identification
def identify_need(input_data: List[str]):
    need_vectors = get_embeddings(input_data)
    points = [
        {
            "id": i, 
            "vector": vector, 
            "payload": {"need": text, "type": "need"}
        } 
        for i, (vector, text) in enumerate(zip(need_vectors, input_data))
    ]
    qdrant.upsert(collection_name="nexus_invention", points=points)
    return np.mean(need_vectors, axis=0)  # Average need vector

# Ideation
def generate_solution(need_vector, criteria_weights: dict = {"feasibility": 0.4, "impact": 0.4, "cost": 0.2}):
    solutions = []
    for i in range(3):
        # Perturb need vector for variety
        solution_vector = need_vector + np.random.normal(0, 0.1, len(need_vector))
        # Mock scoring based on criteria
        score = sum(w * np.random.uniform(0.7, 1.0) for w in criteria_weights.values())
        solutions.append({
            "vector": solution_vector, 
            "score": score, 
            "idea": f"Solution_{i}",
            "description": f"AI-generated solution based on need vector with score {score:.3f}"
        })
    
    # Store all solutions
    points = [
        {
            "id": 100 + i, 
            "vector": s["vector"], 
            "payload": {
                "idea": s["idea"],
                "score": s["score"],
                "description": s["description"],
                "type": "solution"
            }
        } 
        for i, s in enumerate(solutions)
    ]
    qdrant.upsert(collection_name="nexus_invention", points=points)
    
    # Return best solution
    return max(solutions, key=lambda x: x["score"])

# Improvement
def improve_solution(solution, feedback: List[str]):
    # Analyze feedback sentiment
    feedback_scores = [0.2 if "positive" in f.lower() else -0.1 for f in feedback]
    feedback_vector = np.mean(feedback_scores) * np.ones_like(solution["vector"])
    
    improved_vector = solution["vector"] + feedback_vector
    improved_score = solution["score"] + np.mean(feedback_scores) * 0.3
    
    return {
        "vector": improved_vector, 
        "idea": solution["idea"] + "_improved",
        "score": improved_score,
        "description": f"Improved version of {solution['idea']} based on feedback"
    }

# Validation
def validate_invention(improved_solution, metrics: dict = {"success_rate": 0.8, "latency": 200}):
    # Mock validation against existing solutions
    existing_solutions = qdrant.scroll(collection_name="nexus_invention", limit=10)[0]
    if not existing_solutions:
        return {"valid": True, "score": improved_solution["score"]}
    
    existing_vectors = [point.vector for point in existing_solutions]
    similarities = np.dot(existing_vectors, improved_solution["vector"]) / (
        np.linalg.norm(existing_vectors, axis=1) * np.linalg.norm(improved_solution["vector"])
    )
    
    avg_similarity = np.mean(similarities)
    validity = avg_similarity < 0.7  # Not too similar to existing solutions
    
    return {"valid": validity, "score": improved_solution["score"] * (0.5 + avg_similarity / 2)}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/invent")
def invent_process(needs: List[str], feedback: Optional[List[str]] = None):
    need_vector = identify_need(needs)
    solution = generate_solution(need_vector)
    
    if feedback:
        improved = improve_solution(solution, feedback)
        result = validate_invention(improved)
        
        # Store improved solution
        qdrant.upsert(
            collection_name="nexus_invention", 
            points=[{
                "id": 200, 
                "vector": improved["vector"], 
                "payload": {
                    "idea": improved["idea"],
                    "score": improved["score"],
                    "valid": result["valid"],
                    "type": "improved_solution"
                }
            }]
        )
        
        return {
            "idea": improved["idea"],
            "description": improved["description"],
            "valid": result["valid"],
            "score": result["score"]
        }
    
    return {
        "idea": solution["idea"],
        "description": solution["description"],
        "score": solution["score"]
    }