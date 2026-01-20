# BERT Pool Container - Pure Resource Allocation
import modal
from datetime import datetime

image = modal.Image.debian_slim().pip_install("fastapi", "uvicorn", "psutil")
app = modal.App("bert-pool", image=image)

# Always-On BERT Cores
@app.function(memory=1024, cpu=1.0, schedule=modal.Cron("* * * * *"))
def bert_always_on_1():
    print(f"BERT-1 ACTIVE: {datetime.now().isoformat()}")
    return {"bert_id": "bert-1", "cpu": 1.0, "status": "available"}

@app.function(memory=1024, cpu=1.0, schedule=modal.Cron("* * * * *"))
def bert_always_on_2():
    print(f"BERT-2 ACTIVE: {datetime.now().isoformat()}")
    return {"bert_id": "bert-2", "cpu": 1.0, "status": "available"}

# On-Demand BERT Cores
@app.function(memory=2048, cpu=2.0, timeout=3600)
def bert_on_demand():
    print(f"ON-DEMAND BERT: {datetime.now().isoformat()}")
    return {"bert_id": "bert-demand", "cpu": 2.0, "status": "processing"}

# GPU BERT Cores
@app.function(gpu="T4", memory=4096, timeout=1800)
def bert_gpu():
    print(f"GPU BERT: {datetime.now().isoformat()}")
    return {"bert_id": "bert-gpu", "gpu": "T4", "status": "heavy_lifting"}

# BERT Pool API
@app.function(memory=1024)
@modal.asgi_app()
def bert_pool_api():
    from fastapi import FastAPI
    
    api = FastAPI(title="BERT Pool")
    
    @api.get("/")
    def pool_status():
        return {
            "service": "BERT Pool",
            "always_on_berts": 2,
            "on_demand_berts": 8,
            "gpu_berts": 4,
            "status": "ACTIVE"
        }
    
    @api.post("/allocate_cpu")
    def allocate_cpu_bert():
        bert_on_demand.spawn()
        return {"allocated": "cpu_bert", "cores": 2}
    
    @api.post("/allocate_gpu")
    def allocate_gpu_bert():
        bert_gpu.spawn()
        return {"allocated": "gpu_bert", "type": "T4"}
    
    return api

if __name__ == "__main__":
    modal.run(app)