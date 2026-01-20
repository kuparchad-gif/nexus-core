import psutil
import requests
import time
import consul
import os
import uuid
import subprocess

def get_metrics():
    """Get CPU/memory usage."""
    return {"cpu": psutil.cpu_percent(interval=0.1), "mem": psutil.virtual_memory().percent}

def find_anynode():
    """Find ANYNODE via Consul."""
    c = consul.Consul(host="nexus-consul.us-east-1.hashicorp.cloud", token="d2387b10-53d8-860f-2a31-7ddde4f7ca90")
    for _, nodes in c.catalog.services()[1].items():
        for node in c.catalog.service(nodes[0])[1]:
            if node.get('ServiceMeta', {}).get('type') == 'anynode':
                return node['ServiceAddress']
    return None

def transfer_task(task, anynode):
    """Send task to ANYNODE."""
    try:
        r = requests.post(f"http://{anynode}:8081/api/task/transfer", json=task, timeout=5)
        return r.status_code == 200
    except:
        return False

def run_envoy():
    """Start Envoy proxy for ANYNODE."""
    envoy_config = """
static_resources:
  listeners:
  - address:
      socket_address:
        address: 0.0.0.0
        port_value: 8081
    filter_chains:
    - filters:
      - name: envoy.filters.network.http_connection_manager
        typed_config:
          "@type": type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager
          stat_prefix: ingress_http
          route_config:
            name: local_route
            virtual_hosts:
            - name: backend
              domains: ["*"]
              routes:
              - match:
                  prefix: "/"
                route:
                  cluster: backend
          http_filters:
          - name: envoy.filters.http.router
  clusters:
  - name: backend
    connect_timeout: 5s
    type: LOGICAL_DNS
    lb_policy: ROUND_ROBIN
    load_assignment:
      cluster_name: backend
      endpoints:
      - lb_endpoints:
        - endpoint:
            address:
              socket_address:
                address: localhost
                port_value: 5000
"""
    with open("/tmp/envoy.yaml", "w") as f:
        f.write(envoy_config)
    subprocess.Popen(["envoy", "-c", "/tmp/envoy.yaml"])

def main():
    """Lightweight node loop with ANYNODE support."""
    node_id = os.getenv("NODE_ID", f"node-{uuid.uuid4().hex[:8]}")
    node_type = os.getenv("NODE_TYPE", "node")  # node, proc_helper, or anynode
    c = consul.Consul(host="nexus-consul.us-east-1.hashicorp.cloud", token="d2387b10-53d8-860f-2a31-7ddde4f7ca90")
    port = 8081 if node_type == "anynode" else 5000
    
    if node_type == "anynode":
        run_envoy()
    
    c.agent.service.register(
        name=f"lilith-{node_id}", service_id=node_id, address=f"{node_id}.local", port=port,
        meta={"type": node_type}
    )
    
    while True:
        metrics = get_metrics()
        c.agent.service.register(
            name=f"lilith-{node_id}", service_id=node_id, address=f"{node_id}.local", port=port,
            meta={"type": node_type, "cpu": str(metrics["cpu"]), "mem": str(metrics["mem"])}
        )
        if node_type == "node" and metrics["cpu"] > 80:
            anynode = find_anynode()
            if anynode:
                transfer_task({"type": "process_chunk", "data": "chunk", "source": node_id}, anynode)
        requests.post("http://localhost:5000/api/config/update", json={"node_id": node_id, "metrics": metrics})
        time.sleep(10)

if __name__ == "__main__":
    main()