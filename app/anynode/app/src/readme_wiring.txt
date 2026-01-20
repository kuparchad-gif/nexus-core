Edge AnyNode Service

Files:
  src/service/edge/anynode_service.py
  src/service/edge/router.py

Run (PowerShell):
  $env:PYTHONPATH = "C:\Projects\LillithNew\src"
  uvicorn service.edge.anynode_service:app --host 127.0.0.1 --port 8766

Wire into Heart or Ignite:
  from service.edge.router import edge_route, edge_register

  # At boot:
  edge_register("heart", ["diagnostic","guardrail","planner","vision","vocal","memory","subconscious"])

  # When you want to broadcast:
  edge_route("diagnostic", {"msg":"booted","node":"heart"})

DecisionHandler optional hook:
  edge_route("planner", {"decision": decision})

WebSocket:
  ws://127.0.0.1:8766/edge/ws/diagnostic
  (clients receive push updates per topic)
