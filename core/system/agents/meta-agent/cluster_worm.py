# bookworms/cluster_worm.py
import os, json, time, socket
def collect_cluster_nodes():
    # In production, this would query the Orc/portal or a service registry.
    # For now, record the local node only.
    return [{
        "node_id": socket.gethostname(),
        "role": "archiver",
        "ip": "127.0.0.1",
        "status": "active",
        "labels": json.dumps(["primary"]),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }]
