# ai_learning_teacher.py
import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI
from .core import qdrant, get_embedding
from typing import Optional

app = FastAPI()

# Supervised Learning (Linear Regression)
def supervised_learn(data: np.ndarray, labels: np.ndarray):
    data_tensor = torch.tensor(data, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    
    model = nn.Linear(data.shape[1], 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    for epoch in range(100):
        outputs = model(data_tensor)
        loss = criterion(outputs, labels_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return model, loss.item()

# Unsupervised Learning (K-Means)
def unsupervised_learn(data: np.ndarray, k=3):
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    for _ in range(10):
        distances = np.array([[np.linalg.norm(x-c) for c in centroids] for x in data])
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
        if np.all(centroids == new_centroids): 
            break
        centroids = new_centroids
    
    # Calculate inertia as loss proxy
    inertia = sum(np.min(distances, axis=1))
    return labels, centroids, inertia

# Reinforcement Learning (Q-Learning) - Simplified for demo
def reinforcement_learn(states: int, actions: int, rewards: list):
    q_table = np.zeros((states, actions))
    total_reward = 0
    for episode in range(100):
        state = np.random.randint(0, states)
        episode_reward = 0
        for step in range(10):
            action = np.argmax(q_table[state]) if np.random.random() > 0.1 else np.random.randint(actions)
            reward = rewards[state][action]
            q_table[state][action] += 0.1 * (reward + 0.9 * np.max(q_table[action]) - q_table[state][action])
            episode_reward += reward
            state = action
        total_reward += episode_reward
    return q_table, -total_reward  # Negative reward as loss

# Meta-Learning to Choose Best Method
def meta_learn(data: np.ndarray, labels: Optional[np.ndarray] = None):
    methods = []
    losses = []
    
    # Try supervised if labels exist
    if labels is not None:
        try:
            model, loss = supervised_learn(data, labels)
            methods.append("supervised")
            losses.append(loss)
        except:
            pass
    
    # Try unsupervised
    try:
        _, _, loss = unsupervised_learn(data)
        methods.append("unsupervised")
        losses.append(loss)
    except:
        pass
    
    # Try reinforcement with mock rewards
    try:
        rewards = np.random.rand(data.shape[0], 3)  # Mock rewards
        _, loss = reinforcement_learn(data.shape[0], 3, rewards)
        methods.append("reinforcement")
        losses.append(loss)
    except:
        pass
    
    if not methods:
        return "unsupervised", float('inf')
    
    best_method = methods[np.argmin(losses)]
    return best_method, min(losses)

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/learn")
def teach_ai(data: dict):
    features = np.array(data["features"])
    labels = np.array(data["labels"]) if "labels" in data else None
    
    best_method, loss = meta_learn(features, labels)
    
    # Store result in Qdrant
    result_vector = get_embedding(f"{best_method}_{loss}")
    qdrant.upsert(
        collection_name="nexus_learn",
        points=[{
            "id": 1, 
            "vector": result_vector, 
            "payload": {"method": best_method, "loss": loss, "type": "learned"}
        }]
    )
    
    return {"status": "learned", "method": best_method, "loss": loss}