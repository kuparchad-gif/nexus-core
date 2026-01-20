from modal import App, Image, web_endpoint

app = App("lillith-repair")

node_image = Image.debian_slim().pip_install("fastapi", "uvicorn", "grpcio", "consul", "web3", "stripe", "twilio")
python_image = Image.debian_slim().pip_install("fastapi", "uvicorn", "grpcio", "consul", "web3", "stripe", "twilio", "transformers")

@app.function(image=node_image, cpu=2, memory=4096, disk=20*1024)
@web_endpoint()
def web_interface():
    import subprocess
    subprocess.run(["node", "it-pro-diag.js"])
    return {"status": "Web interface running on port 5003"}

@app.function(image=python_image, cpu=2, memory=4096, disk=20*1024)
def swarm_manager():
    import swarm_manager
    swarm_manager.start()

@app.function(image=python_image, cpu=2, memory=4096, disk=20*1024)
def llm_loader():
    import llm_loader
    llm_loader.start()

@app.function(image=python_image, cpu=2, memory=4096, disk=20*1024)
def viren_core():
    import viren_core
    viren_core.start()

@app.local_entrypoint()
def main():
    web_interface.deploy()
    swarm_manager.deploy()
    llm_loader.deploy()
    viren_core.deploy()