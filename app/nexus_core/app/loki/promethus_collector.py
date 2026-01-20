# prometheus_collector.py
import modal
from fastapi import FastAPI
import os

app = modal.App("prometheus-collector")
image = modal.Image.debian_slim().pip_install("prom/prometheus")  # Official Prometheus

prom_vol = modal.Volume.from_name("prometheus-data", create_if_missing=True)

# Prometheus config (scrapes your apps every 15s)
prom_config = """
global:
  scrape_interval: 15s
scrape_configs:
  - job_name: 'aries'
    static_configs:
      - targets: ['aries-funding.modal.app:8000']  # Internal Modal DNS
    metrics_path: /metrics
  - job_name: 'decision'
    static_configs:
      - targets: ['decision-engine.modal.app:8000']
    metrics_path: /metrics
  # Add one per app (12 total)
"""

@app.function(image=image, volumes={"/prometheus": prom_vol})
def run_prometheus():
    with open("/prometheus/prometheus.yml", "w") as f:
        f.write(prom_config)
    os.system("prometheus --config.file=/prometheus/prometheus.yml --storage.tsdb.path=/prometheus/data --web.enable-lifecycle")
    return {"status": "prometheus_running"}

@app.function(image=image)
@modal.web_server(9090)  # Prometheus UI
def prometheus_ui():
    web = FastAPI()
    @web.get("/")
    async def root():
        return {"prometheus": "alive", "scrape_url": "http://localhost:9090/metrics"}
    return web