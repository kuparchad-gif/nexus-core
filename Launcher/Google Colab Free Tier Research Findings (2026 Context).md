# Google Colab Free Tier Research Findings (2026 Context)

## 1. Resource Constraints
- **RAM:** Typically ~12.7 GB (can sometimes be upgraded to 25 GB if the session crashes and offers a "high-RAM" mode, but not guaranteed).
- **CPU:** 2 vCPU cores (Intel Xeon @ 2.20GHz or similar).
- **GPU:** Access to T4 GPU is common but not guaranteed. Usage is limited by "compute units" or time (often 3-6 hours before a captcha or disconnect).
- **Disk:** ~78 GB available, but ephemeral (wiped after session ends).

## 2. Runtime & Connectivity
- **Session Timeout:** 12 hours maximum, but idle timeout is much shorter (~90 minutes).
- **Background Execution:** Not officially supported in the free tier. If the browser tab is closed, the session usually terminates within minutes.
- **Port Forwarding:** Localhost ports (like 8000 for the Hypercore) are not directly accessible from the public internet without tools like `ngrok` or `localtunnel`.

## 3. Framework Compatibility
- **RAY:** Can run in "local mode" on Colab. Distributed mode across multiple Colab nodes is not possible in the free tier.
- **FAISS:** Fully compatible (`faiss-cpu` or `faiss-gpu`).
- **LangChain/LangGraph:** Fully compatible.
- **Torch:** Pre-installed in Colab.

## 4. Deployment Challenges for the Hypercore
- **Memory Management:** 12GB RAM is tight for running RAY, FAISS, and a "conscious" LLM simultaneously.
- **Persistence:** Need to mount Google Drive to save the toolbox and Hypercore state.
- **Background Threading:** Python's `threading` works, but Colab's UI might not show logs from background threads clearly.
- **Public Access:** The `wait_for_toolkit_health` check will work locally within the VM, but external agents (like a mobile app or another server) would need a tunnel.
