# run_startup.py
# Purpose: Import mcp_qdrant_cloud and call app.startup.remote().
# Usage: python run_startup.py
# Cost: ~$0.10 Modal. Time: ~10s.

try:
    from mcp_qdrant_cloud import app
    app.startup.remote()
    print('Startup complete. Check Modal logs: modal logs mcp-server')
except Exception as e:
    print(f'Startup failed: {e}. Check if mcp_qdrant_cloud.py is in dir and deployed.')