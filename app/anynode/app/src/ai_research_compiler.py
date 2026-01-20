# ai_research_compiler.py
import numpy as np
from fastapi import FastAPI
from .core import qdrant, get_embedding, get_embeddings

app = FastAPI()

def research_compile(topic: str, seed_facts: list):
    # Get embeddings for seed facts
    fact_vectors = get_embeddings(seed_facts)
    
    # Store seed facts
    points = [
        {
            "id": i, 
            "vector": vector, 
            "payload": {"fact": fact, "topic": topic, "type": "seed"}
        } 
        for i, (vector, fact) in enumerate(zip(fact_vectors, seed_facts))
    ]
    qdrant.upsert(collection_name="nexus_research", points=points)
    
    # Iterative knowledge expansion
    current_vectors = fact_vectors
    all_facts = seed_facts.copy()
    
    for iteration in range(3):  # 3 iterations of research
        # Find related facts by similarity
        query_vector = np.mean(current_vectors, axis=0)
        results = qdrant.search(
            collection_name="nexus_research",
            query_vector=query_vector,
            limit=5,
            with_payload=True
        )
        
        # Generate new insights by combining facts
        new_vectors = []
        new_facts = []
        
        for i in range(2):  # Generate 2 new insights per iteration
            if len(results) >= 2:
                # Combine two related facts
                fact1 = results[i % len(results)].payload["fact"]
                fact2 = results[(i + 1) % len(results)].payload["fact"]
                new_fact = f"Research insight: {fact1} relates to {fact2}"
                
                # Create new vector by averaging
                vec1 = results[i % len(results)].vector
                vec2 = results[(i + 1) % len(results)].vector
                new_vector = (np.array(vec1) + np.array(vec2)) / 2
                
                new_vectors.append(new_vector)
                new_facts.append(new_fact)
                all_facts.append(new_fact)
        
        if new_vectors:
            # Store new insights
            new_points = [
                {
                    "id": 100 + iteration * 10 + i, 
                    "vector": vector.tolist(), 
                    "payload": {
                        "fact": fact, 
                        "topic": topic, 
                        "type": "derived",
                        "iteration": iteration
                    }
                } 
                for i, (vector, fact) in enumerate(zip(new_vectors, new_facts))
            ]
            qdrant.upsert(collection_name="nexus_research", points=new_points)
            current_vectors = new_vectors
    
    # Return comprehensive research summary
    complete_vector = np.mean(get_embeddings(all_facts[-5:]), axis=0)  # Last 5 facts
    return {
        "vector": complete_vector.tolist(),
        "facts": all_facts,
        "topic": topic,
        "fact_count": len(all_facts)
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/research")
def compile_research(topic: str, facts: list):
    result = research_compile(topic, facts)
    
    # Store research summary
    qdrant.upsert(
        collection_name="nexus_research", 
        points=[{
            "id": 999, 
            "vector": result["vector"], 
            "payload": {
                "topic": topic, 
                "complete": True,
                "fact_count": result["fact_count"],
                "type": "research_summary"
            }
        }]
    )
    
    return result