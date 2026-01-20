# Cloud Viren Build Instructions

This document provides instructions for building and deploying Cloud Viren with Weaviate integration for technical knowledge, problem-solving concepts, and troubleshooting tools.

## Prerequisites

- Docker and Docker Compose
- Python 3.8 or higher
- At least 16GB RAM and 4 CPU cores
- 100GB disk space
- Network access for pulling Docker images and Python packages

## Step 1: Set Up the Environment

1. Clone the repository or extract the files to your desired location
2. Navigate to the Viren directory:

```bash
cd C:\Viren
```

3. Create a Python virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

4. Install the required packages:

```bash
pip install -r requirements.txt
```

## Step 2: Configure Weaviate

1. Update the Weaviate configuration file:

```bash
# Edit C:\Viren\vector\weaviate_config.json to match your environment
```

2. Start Weaviate using Docker:

```bash
docker pull weaviate/weaviate
docker run -d -p 8080:8080 -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true -e PERSISTENCE_DATA_PATH=/var/lib/weaviate -v weaviate_data:/var/lib/weaviate weaviate/weaviate
```

3. Verify Weaviate is running:

```bash
curl http://localhost:8080/v1/.well-known/ready
```

## Step 3: Initialize the Schema

1. Create a Python script to initialize the Weaviate schema:

```bash
# Create C:\Viren\scripts\init_weaviate.py
```

```python
import weaviate
from vector.weaviate_schema import create_weaviate_schema

# Connect to Weaviate
client = weaviate.Client("http://localhost:8080")

# Create schema
create_weaviate_schema(client)

print("Schema created successfully")
```

2. Run the script:

```bash
python scripts\init_weaviate.py
```

## Step 4: Prepare Sample Data

1. Create directories for sample data:

```bash
mkdir -p C:\Viren\data\technical_docs
```

2. Create sample problem-solving concepts:

```bash
# Create C:\Viren\data\problem_solving_concepts.json
```

```json
[
  {
    "name": "Root Cause Analysis",
    "description": "A systematic approach to identifying the underlying cause of a problem rather than treating symptoms.",
    "applicability": ["Troubleshooting", "Debugging", "System Optimization"],
    "steps": [
      "Define the problem clearly",
      "Collect data and evidence",
      "Identify possible causal factors",
      "Identify the root cause",
      "Recommend and implement solutions"
    ],
    "examples": [
      "Identifying memory leaks in a long-running service",
      "Diagnosing intermittent network connectivity issues",
      "Resolving database performance degradation"
    ],
    "relatedConcepts": ["5 Whys Technique", "Fishbone Diagram", "Fault Tree Analysis"]
  },
  {
    "name": "5 Whys Technique",
    "description": "An iterative interrogative technique used to explore the cause-and-effect relationships underlying a particular problem.",
    "applicability": ["Troubleshooting", "Process Improvement", "Quality Control"],
    "steps": [
      "State the problem clearly",
      "Ask why the problem occurs and write down the answer",
      "If the answer doesn't identify the root cause, ask why again",
      "Repeat until the root cause is identified (typically five times)",
      "Develop countermeasures to prevent recurrence"
    ],
    "examples": [
      "Why is the server down? Because the CPU usage is at 100%.",
      "Why is the CPU usage at 100%? Because there are too many requests being processed.",
      "Why are there too many requests? Because the rate limiting is not working.",
      "Why is rate limiting not working? Because the configuration was overridden.",
      "Why was the configuration overridden? Because the deployment process didn't validate changes."
    ],
    "relatedConcepts": ["Root Cause Analysis", "Pareto Analysis", "PDCA Cycle"]
  }
]
```

3. Create sample troubleshooting tools:

```bash
# Create C:\Viren\data\troubleshooting_tools.json
```

```json
[
  {
    "name": "Network Packet Analyzer",
    "description": "A tool for capturing and analyzing network packets to diagnose connectivity and performance issues.",
    "usage": "Run the tool with administrator privileges and specify the network interface to capture packets from.",
    "parameters": [
      "-i, --interface: Network interface to capture packets from",
      "-f, --filter: BPF filter expression to filter packets",
      "-o, --output: File to save captured packets",
      "-c, --count: Number of packets to capture"
    ],
    "outputFormat": "PCAP file or text summary of captured packets",
    "category": "Networking",
    "compatibleSystems": ["Windows", "Linux", "macOS"]
  },
  {
    "name": "Memory Profiler",
    "description": "A tool for analyzing memory usage and identifying memory leaks in applications.",
    "usage": "Attach the profiler to a running process or start a new process with the profiler.",
    "parameters": [
      "-p, --pid: Process ID to attach to",
      "-c, --command: Command to run with profiling",
      "-i, --interval: Sampling interval in milliseconds",
      "-o, --output: File to save profiling data"
    ],
    "outputFormat": "HTML report with memory usage graphs and allocation stacks",
    "category": "Performance",
    "compatibleSystems": ["Windows", "Linux"]
  }
]
```

## Step 5: Index Sample Data

1. Create a Python script to index the sample data:

```bash
# Create C:\Viren\scripts\index_sample_data.py
```

```python
import os
import json
from vector.weaviate_client import CloudVirenWeaviateClient
from vector.weaviate_indexer import WeaviateIndexer

# Connect to Weaviate
client = CloudVirenWeaviateClient("http://localhost:8080")

# Create indexer
indexer = WeaviateIndexer(client)

# Index problem-solving concepts
concepts_file = "C:/Viren/data/problem_solving_concepts.json"
if os.path.exists(concepts_file):
    processed, success, errors = indexer.index_problem_solving_concepts(concepts_file)
    print(f"Indexed {success} problem-solving concepts with {errors} errors")

# Index troubleshooting tools
tools_file = "C:/Viren/data/troubleshooting_tools.json"
if os.path.exists(tools_file):
    processed, success, errors = indexer.index_troubleshooting_tools(tools_file)
    print(f"Indexed {success} troubleshooting tools with {errors} errors")

# Index technical documentation
docs_dir = "C:/Viren/data/technical_docs"
if os.path.exists(docs_dir):
    processed, success, errors = indexer.index_technical_documentation(docs_dir)
    print(f"Indexed {success} technical documents with {errors} errors")

print("Indexing complete")
```

2. Run the script:

```bash
python scripts\index_sample_data.py
```

## Step 6: Start the Vector MCP

1. Create a startup script:

```bash
# Create C:\Viren\scripts\start_vector_mcp.py
```

```python
import sys
import os

# Add the Viren directory to the Python path
sys.path.insert(0, os.path.abspath("C:/Viren"))

from vector.vector_mcp import VectorMCP
from api.mcp_api import start_api

# Create and start the Vector MCP
mcp = VectorMCP()
mcp.start()

# Start the API
start_api(mcp, port=8000)
```

2. Run the script:

```bash
python scripts\start_vector_mcp.py
```

## Step 7: Configure Cloud Integration

1. Update the cloud configuration:

```bash
# Edit C:\Viren\cloud\cloud_connection.py to match your environment
```

2. Start the cloud connection:

```bash
python -c "from cloud.cloud_connection import CloudConnection; CloudConnection().start()"
```

## Step 8: Deploy with Docker Compose

1. Update the Docker Compose file:

```bash
# Edit C:\Viren\data\docker-compose.yml
```

```yaml
version: '3'

services:
  weaviate:
    image: weaviate/weaviate:latest
    ports:
      - "8080:8080"
    environment:
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
    volumes:
      - weaviate_data:/var/lib/weaviate
    restart: unless-stopped

  vector-mcp:
    build:
      context: ..
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    depends_on:
      - weaviate
    environment:
      WEAVIATE_URL: 'http://weaviate:8080'
    volumes:
      - ../data:/app/data
      - ../logs:/app/logs
    restart: unless-stopped

volumes:
  weaviate_data:
```

2. Create a Dockerfile:

```bash
# Create C:\Viren\Dockerfile
```

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "scripts/start_vector_mcp.py"]
```

3. Start the services:

```bash
cd C:\Viren\data
docker-compose up -d
```

## Step 9: Verify the Deployment

1. Check if the services are running:

```bash
docker-compose ps
```

2. Test the API:

```bash
curl http://localhost:8000/health
```

3. Open the API documentation:

```
http://localhost:8000/docs
```

## Step 10: Next Steps

1. Add more technical documentation to `C:\Viren\data\technical_docs`
2. Expand the problem-solving concepts in `C:\Viren\data\problem_solving_concepts.json`
3. Add more troubleshooting tools to `C:\Viren\data\troubleshooting_tools.json`
4. Integrate with the model cascade for advanced reasoning
5. Set up monitoring and alerting for the cloud deployment

## Troubleshooting

### Weaviate Connection Issues

If you can't connect to Weaviate:

```bash
# Check if Weaviate is running
docker ps | grep weaviate

# Check Weaviate logs
docker logs $(docker ps -q --filter "name=weaviate")

# Restart Weaviate
docker restart $(docker ps -q --filter "name=weaviate")
```

### API Connection Issues

If you can't connect to the API:

```bash
# Check if the API is running
curl http://localhost:8000/health

# Check API logs
docker logs $(docker ps -q --filter "name=vector-mcp")

# Restart the API
docker restart $(docker ps -q --filter "name=vector-mcp")
```

### Data Indexing Issues

If data isn't being indexed correctly:

```bash
# Check the indexer logs
cat C:\Viren\logs\weaviate.log

# Verify the data files exist
dir C:\Viren\data

# Run the indexing script with debug logging
python -m vector.weaviate_indexer --debug
```