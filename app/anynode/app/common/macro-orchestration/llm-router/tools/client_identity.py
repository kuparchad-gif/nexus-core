```python
import asyncio
import json
import logging
import ssl
from nats.aio.client import Client as NATS
import requests
import os
import socket

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IdentityClient:
    def __init__(self):
        self.nats_url = "tls://[REDACTED-IP]:4222"
        self.nats_ca = "/var/lib/metatron/pki/ca/ca.crt"
        self.nats_cert = "/var/lib/metatron/pki/server/server.crt"
        self.nats_key = "/var/lib/metatron/pki/server/server.key"
        self.loki_url = "[REDACTED-URL]
        self.container_id = os.getenv("HOSTNAME", socket.gethostname())
        self.pod_name = os.getenv("POD_NAME", "unknown")
        self.role = os.getenv("ROLE", "unknown")  # Set via container env or logic
        self.ports = os.getenv("PORTS", "")  # Set via container env or inspection

    async def connect_nats(self):
        """Connect to NATS."""
        self.nc = NATS()
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ssl_context.load_verify_locations(self.nats_ca)
        ssl_context.load_cert_chain(self.nats_cert, self.nats_key)
        await self.nc.connect(self.nats_url, tls=ssl_context)
        logger.info("Connected to NATS")

    def log_to_loki(self, message):
        """Send log to Loki."""
        try:
            requests.post(self.loki_url, json={
                "streams": [{
                    "stream": {"app": "client", "container": self.container_id},
                    "values": [[str(int(1e9 * time.time())), message]]
                }]
            })
        except requests.RequestException as e:
            logger.error(f"Failed to log to Loki: {e}")

    async def check_in(self):
        """Register with archivist."""
        metadata = {
            "container_id": self.container_id,
            "pod_name": self.pod_name,
            "role": self.role,
            "ports": self.ports,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        try:
            response = await self.nc.request(
                "mcp.nexus-core.checkin",
                json.dumps(metadata).encode(),
                timeout=5
            )
            result = json.loads(response.data.decode())
            logger.info(f"Check-in response: {result}")
            self.log_to_loki(f"Checked in: {result}")
        except Exception as e:
            logger.error(f"Check-in failed: {e}")
            self.log_to_loki(f"Check-in failed: {e}")

    async def query_identity(self):
        """Query archivist for identity and instructions."""
        try:
            response = await self.nc.request(
                "mcp.nexus-core.query",
                json.dumps({"container_id": self.container_id}).encode(),
                timeout=5
            )
            result = json.loads(response.data.decode())
            logger.info(f"Identity response: {result}")
            self.log_to_loki(f"Identity query: {result}")
            return result
        except Exception as e:
            logger.error(f"Query failed: {e}")
            self.log_to_loki(f"Query failed: {e}")
            return {"identity": None, "instruction": "Failed to query"}

    async def run(self):
        """Run check-in and query."""
        await self.connect_nats()
        await self.check_in()
        result = await self.query_identity()
        await self.nc.close()
        return result

if __name__ == "__main__":
    import time
    client = IdentityClient()
    result = asyncio.run(client.run())
    print(f"Identity: {result.get('identity')}")
    print(f"Instruction: {result.get('instruction')}")
```
