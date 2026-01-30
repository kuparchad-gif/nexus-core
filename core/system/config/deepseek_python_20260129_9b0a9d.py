# 1. Mount Google Drive (optional for persistence)
from google.colab import drive
drive.mount('/content/drive')

# 2. Set secrets
import os
os.environ['HF_TOKEN'] = 'your_huggingface_token_here'

# 3. Install dependencies
!pip install torch transformers ray fastapi uvicorn pymongo numpy scipy huggingface-hub qdrant-client

# 4. Run the orchestrator
import asyncio
from ultimate_colab_orchestrator import UltimateColabOrchestrator

orchestrator = UltimateColabOrchestrator()

# Initialize system
await orchestrator.initialize()

# Run full pipeline
result = await orchestrator.run_full_pipeline("https://github.com/yourusername/your-repo.git")