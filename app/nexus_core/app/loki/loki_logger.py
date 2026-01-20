# loki_logger.py
import modal
from fastapi import FastAPI

app = modal.App("loki-logger")
image = modal.Image.debian_slim().pip_install("grafana/loki")  # Official Loki

loki_vol = modal.Volume.from_name("loki-data", create_if_missing=True)

@app.function(image=image, volumes={"/loki": loki_vol})
def run_loki():
    os.system("loki -config.file=/loki/loki-config.yml")  # Minimal config in volume
    return {"status": "loki_running"}

# Simple config (create on first run)
loki_config = """
auth_enabled: false
server:
  http_listen_port: 3100
ingester:
  lifecycler:
    address: 127.0.0.1
    ring:
      kvstore:
        store: inmemory
      replication_factor: 1
    final_sleep: 0s
  chunk_idle_period: 5m
  chunk_retain_period: 30s
schema_config:
  configs:
    - from: 2020-05-15
      store: boltdb-shipper
      object_store: filesystem
      schema: v11
      index:
        prefix: index_
        period: 168h
storage_config:
  boltdb_shipper:
    active_index_directory: /loki/index
    cache_location: /loki/index_cache
    shared_store: filesystem
  filesystem:
    directory: /loki/chunks
limits_config:
  enforce_metric_name: false
  reject_old_samples: true
  reject_old_samples_max_age: 168h
chunk_store_config:
  max_look_back_period: 168h
table_manager:
  retention_deletes_enabled: false
  respect_retention_period: false
compactor:
  working_directory: /data
  shared_store: filesystem
"""

@app.function(image=image)
@modal.web_server(3100)  # Loki push endpoint
def loki_push():
    web = FastAPI()
    @web.post("/loki/api/v1/push")  # Receives log pushes
    async def push_logs(request: Request):
        # Forward to Loki (or handle in function)
        return {"received": True}
    return web