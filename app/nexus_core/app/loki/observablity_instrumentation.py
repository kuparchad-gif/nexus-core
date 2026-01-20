# === OBSERVABILITY INSTRUMENTATION ===
from prometheus_fastapi_instrumentator import Instrumentator
from logging_loki import LokiHandler
import logging

# Prometheus: Auto-instrument metrics
instrumentator = Instrumentator().instrument(web).expose(web, endpoint="/metrics")  # Scrapable at /metrics

# Loki: Structured logging (push to central Loki)
logger = logging.getLogger(__name__)
loki_handler = LokiHandler(url="http://loki-logger.modal.app:3100/loki/api/v1/push",  # Internal Modal DNS
                           tags={"app": "aries", "service": "funding"})  # Labels for Grafana queries
loki_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
loki_handler.setFormatter(formatter)
logger.addHandler(loki_handler)

# Example: Log + metric in an endpoint
@web.post("/start")
async def start(req: StartReq):
    logger.info("Starting investor conversation", extra={"investor_id": req.investor.get("id")})
    # ... your code ...