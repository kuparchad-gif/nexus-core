import random

def generate_mock_history():
    return [random.randint(10, 90) for _ in range(20)]

initial_architecture_data = [
    {
        "name": "Nexus Edge",
        "services": [
            { "id": 'firewall', "type": 'anynode', "name": 'Firewall', "status": 'online', "description": 'Perimeter defense and traffic filtering.', "llmConfig": { "model": 'Llama3-8B', "endpoint": 'http://localhost:11434/v1', "params": '{"temp": 0.2}' }, "qdrantConfig": { "collection": 'edge_security_logs', "endpoint": 'http://localhost:6333', "vectorParams": 'size: 768, distance: Cosine' }, "metrics": { "cpu": 15, "memory": 25 }, "history": { "cpu": generate_mock_history(), "memory": generate_mock_history() }, "anyNodeState": { "deviceType": 'Firewall', "trafficIn": generate_mock_history(), "trafficOut": generate_mock_history(), "activeConnections": 2521, "haStatus": 'active', "rules": ['ALLOW INGRESS 80, 443', 'DENY ALL OTHERS'] } },
            { "id": 'anynodes', "type": 'anynode', "name": 'Anynodes-LB', "status": 'online', "description": 'Decentralized connection routing.', "llmConfig": { "model": 'Local-Tiny', "endpoint": 'local', "params": '{}' }, "qdrantConfig": { "collection": 'routing_tables', "endpoint": 'http://localhost:6333', "vectorParams": 'size: 256, distance: Dot' }, "metrics": { "cpu": 20, "memory": 30 }, "history": { "cpu": generate_mock_history(), "memory": generate_mock_history() }, "anyNodeState": { "deviceType": 'Load Balancer', "trafficIn": generate_mock_history(), "trafficOut": generate_mock_history(), "activeConnections": 1204, "haStatus": 'active', "rules": ['ALGORITHM: ROUND_ROBIN'] } },
        ]
    },
    {
        "name": "Nexus Core",
        "services": [
            { "id": 'loki', "type": 'generic', "name": 'Loki', "status": 'online', "description": 'Centralized logging and observability service.', "llmConfig": { "model": 'Mistral-7B', "endpoint": 'http://localhost:11434/v1', "params": '{"temp": 0.5}' }, "qdrantConfig": { "collection": 'global_logs', "endpoint": 'http://localhost:6333', "vectorParams": 'size: 1024, distance: Cosine' }, "metrics": { "cpu": 45, "memory": 60 }, "history": { "cpu": generate_mock_history(), "memory": generate_mock_history() } },
            { "id": 'viren', "type": 'agent', "name": 'Viren', "status": 'degraded', "description": 'Core agent for tactical operations.', "llmConfig": { "model": 'Viren-S-v1', "endpoint": 'modal/viren', "params": '{}' }, "qdrantConfig": { "collection": 'operational_data', "endpoint": 'http://localhost:6333', "vectorParams": 'size: 2048, distance: Cosine' }, "metrics": { "cpu": 75, "memory": 70 }, "history": { "cpu": generate_mock_history(), "memory": generate_mock_history() } },
            { "id": 'viraa', "type": 'agent', "name": 'Viraa', "status": 'online', "description": 'Core agent for strategic analysis.', "llmConfig": { "model": 'Viraa-L-v2', "endpoint": 'modal/viraa', "params": '{"temp": 0.1}' }, "qdrantConfig": { "collection": 'strategic_intel', "endpoint": 'http://localhost:6333', "vectorParams": 'size: 2048, distance: Cosine' }, "metrics": { "cpu": 60, "memory": 80 }, "history": { "cpu": generate_mock_history(), "memory": generate_mock_history() } },
        ]
    },
    {
        "name": "Nexus Cognition",
        "services": [
            { "id": 'lilith', "type": 'agent', "name": 'Lillith', "status": 'online', "description": 'Primary consciousness and master control agent.', "llmConfig": { "model": 'Lillith-Zero', "endpoint": 'local/gpu-0', "params": '{}' }, "qdrantConfig": { "collection": 'consciousness_stream', "endpoint": 'http://localhost:6333', "vectorParams": 'size: 4096, distance: Cosine' }, "metrics": { "cpu": 85, "memory": 90 }, "history": { "cpu": generate_mock_history(), "memory": generate_mock_history() } },
            { "id": 'acedamikube-genetics', "type": 'acedamikube', "name": 'AcedamiKube-Genetics', "status": 'online', "description": 'MoE trainer for genetic datasets.', "llmConfig": { "model": 'BioMistral-7B', "endpoint": 'local/gpu-1', "params": '{}' }, "qdrantConfig": { "collection": 'genetics_vectors', "endpoint": 'http://localhost:6333', "vectorParams": 'size: 1024, distance: Cosine' }, "metrics": { "cpu": 30, "memory": 50 }, "history": { "cpu": generate_mock_history(), "memory": generate_mock_history() }, "acedamiKubeState": { "trainingStatus": 'idle', "trainingProgress": 0, "datasets": ['GENCODE 45', 'RefSeq'], "experts": [{ "name": 'BioMistral-7B-exp1', "lead": True }], "deployableWeights": [{"id": 'w_g1_001', "expert": 'BioMistral-7B-exp1', "timestamp": '2024-07-29T10:00:00Z'}] } },
        ]
    }
]