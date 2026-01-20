# invoke_bridge.py
# Purpose: Acts as a bridge for routing between Python and TS routers based on target

import json
from bridge.python_router import handle_command as python_handle

# Placeholder for frontend router call
def send_to_ts(method, args=None):
    print(f"[Bridge] Would call TS method: {method} with args: {args}")
    return {"status": "sent to frontend", "method": method, "args": args}

def invoke_bridge(target: str, method: str, args=None):
    if args is None:
        args = []

    if target == "python":
        return python_handle(method, args)
    elif target == "ts":
        return send_to_ts(method, args)
    else:
        return {"error": f"Invalid target: {target}"}

if __name__ == "__main__":
    # EXAMPLES:
    print(invoke_bridge("python", "refresh_models"))
    print(invoke_bridge("ts", "reloadUI", []))
