## File: C:/Viren/viren_portable.py
```python
"""
viren_portable.py
Location: C:/Viren/
Self-contained USB-deployable technician agent with fallback UI.
"""

import platform
import subprocess
import os

print("[PORTABLE] Viren portable technician launching...")

os_type = platform.system()
print(f"[SYSTEM] Detected OS: {os_type}")

try:
    print("[CHECK] Launching fallback Gradio MCP interface...")
    subprocess.run(["python", "Systems/router/mcp_router_logic.py"])
except Exception as e:
    print("[ERROR] Failed to launch MCP: ", e)

print("[PORTABLE] Technician agent completed.")
```