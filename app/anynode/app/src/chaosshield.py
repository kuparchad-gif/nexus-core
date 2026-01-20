import consul
import requests
import os
import uuid
import time
import smtplib
from email.mime.text import MIMEText

def send_alert(message):
    """Send email alert."""
    try:
        msg = MIMEText(message)
        msg['Subject'] = 'ChaosShield Alert'
        msg['From'] = 'lilith@nexus.ai'
        msg['To'] = 'admin@nexus.ai'
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login('your-email@gmail.com', 'your-password')
            server.send_message(msg)
    except Exception as e:
        print(f"Alert send failed: {e}")

def detect_anomaly(metrics):
    """Detect network anomalies."""
    return metrics.get("cpu", 0) > 90 or metrics.get("mem", 0) > 90

def main():
    node_id = os.getenv("NODE_ID", f"chaosshield-{uuid.uuid4().hex[:8]}")
    c = consul.Consul(host="nexus-consul.us-east-1.hashicorp.cloud", token="d2387b10-53d8-860f-2a31-7ddde4f7ca90")
    c.agent.service.register(
        name=f"chaosshield-{node_id}", service_id=node_id, address=f"{node_id}.local", port=8084,
        meta={"type": "chaosshield"}
    )
    
    while True:
        try:
            for _, nodes in c.catalog.services()[1].items():
                for node in c.catalog.service(nodes[0])[1]:
                    metrics = node.get('ServiceMeta', {})
                    if detect_anomaly(metrics):
                        alert_msg = f"Anomaly detected on {node['ServiceID']}: CPU {metrics.get('cpu')}, Mem {metrics.get('mem')}"
                        send_alert(alert_msg)
                        requests.post("http://localhost:5000/api/firewall", 
                                    json={"rule": {"protocol": "tcp", "port": 8081, "allow": False}}, timeout=5)
        except Exception as e:
            print(f"ChaosShield error: {e}")
        time.sleep(10)

if __name__ == "__main__":
    main()