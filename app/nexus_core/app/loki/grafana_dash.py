# grafana_dashboard.py
import modal
from fastapi import FastAPI

app = modal.App("grafana-dashboard")
image = modal.Image.debian_slim().pip_install("grafana/grafana")  # Official

grafana_vol = modal.Volume.from_name("grafana-data", create_if_missing=True)

# Provision datasources (Prometheus + Loki)
provision_config = """
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    url: http://prometheus-collector.modal.app:9090
    access: proxy
    isDefault: true
  - name: Loki
    type: loki
    url: http://loki-logger.modal.app:3100
    access: proxy
"""

@app.function(image=image, volumes={"/grafana": grafana_vol})
def run_grafana():
    with open("/grafana/provisioning/datasources/datasources.yml", "w") as f:
        f.write(provision_config)
    os.system("grafana-server --homepath=/usr/share/grafana")
    return {"status": "grafana_running", "login": "admin/admin"}  # Change in prod

@app.function(image=image)
@modal.web_server(3000)  # Grafana UI
def grafana_ui():
    web = FastAPI()
    @web.get("/")
    async def root():
        return {"grafana": "alive", "dashboard_url": "http://localhost:3000/dashboards"}
    return web