# Google Colab Deployment Guide: Ultimate AI Toolbox

The **Ultimate AI Toolbox** and **Conscious Quantum Hypercore** can be deployed on Google Colab's free tier. However, due to the 12.7 GB RAM limitation and ephemeral nature of Colab sessions, specific configuration steps are required to ensure stability and persistence.

---

## 1. Environment Preparation

Google Colab provides a powerful but restricted environment. To run the full suite, you must first install the necessary dependencies and mount Google Drive for persistent storage.

### Required Setup Code
Copy and run this in the first cell of your Colab notebook:

```python
# 1. Mount Google Drive for persistence
from google.colab import drive
drive.mount('/content/drive')

# 2. Install essential dependencies
!pip install ray[default] faiss-cpu langchain langgraph fastapi uvicorn psutil networkx scipy aiohttp requests GitPython
# Note: Torch is pre-installed in Colab.
```

---

## 2. Resource Optimization

The free tier of Colab is limited to approximately 12.7 GB of RAM. Running RAY, FAISS, and the Hypercore simultaneously can exceed this limit if not properly configured.

### Memory Management Strategy
The system includes a `ColabOptimizer` that automatically detects the environment and applies the following adjustments:

| Component | Standard Config | Colab Optimized |
|:----------|:----------------|:----------------|
| **Memory Cache** | 4096 MB | 1024 MB |
| **RAY Object Store** | Auto (Large) | 512 MB |
| **Vector Precision** | Float32 | Float16 (Suggested) |
| **Parallel Workers** | CPU Count | CPU Count - 1 |

---

## 3. Persistence and Workspace

Because Colab's local disk is wiped after every session, you must store your toolbox and hypercore state on Google Drive.

### Recommended Directory Structure
Set your workspace path to a folder within your mounted drive:

```python
WORKSPACE_PATH = "/content/drive/MyDrive/ultimate_toolbox"
```

---

## 4. Public Access (Tunnelling)

Colab runs on a private network. To access the Hypercore's MCP or FastAPI server from outside the notebook (e.g., from a mobile app or another server), you will need a tunnelling service like **ngrok**.

### Setting up ngrok
```python
!pip install pyngrok
from pyngrok import ngrok

# Authenticate (Get your token from ngrok.com)
# ngrok.set_auth_token("YOUR_TOKEN")

# Open a tunnel to the Hypercore server
public_url = ngrok.connect(8000)
print(f"Hypercore Public URL: {public_url}")
```

---

## 5. Deployment Script (Colab Version)

Use this modified version of your deployment script to launch the system in Colab:

```python
import sys
import os
from pathlib import Path

# Assuming you uploaded the toolbox to your Drive
TOOLBOX_PATH = "/content/drive/MyDrive/ultimate_toolbox"
sys.path.insert(0, TOOLBOX_PATH)
sys.path.insert(0, os.path.join(TOOLBOX_PATH, "integration"))

from conscious_quantum_hypercore_integration import ConsciousQuantumHypercoreOrchestrator
from colab_optimizer import colab_opt

# Apply optimizations
overrides = colab_opt.optimize_memory()

# Initialize Orchestrator
toolkit_orchestrator = ConsciousQuantumHypercoreOrchestrator()

# Start Server in Background
import threading
toolkit_thread = threading.Thread(target=toolkit_orchestrator.run_server, 
                                   args=("0.0.0.0", 8000), daemon=True)
toolkit_thread.start()

print("🚀 Hypercore is running in the background on Colab!")
```

---

## 6. Limitations to Keep in Mind

- **Idle Timeout:** Colab will disconnect if the browser tab is inactive for ~90 minutes. Keep the tab open or use a "no-sleep" script in the console.
- **Session Duration:** Maximum runtime is 12 hours. You will need to restart the session and re-run the setup daily.
- **GPU Availability:** The free tier does not always guarantee a GPU. The system is optimized to fall back to CPU-only mode automatically.

---

**Manus AI**
