# rabbit_trigger_test.py - Verify White Rabbit (Oz Sync)
import modal, numpy as np
from fastapi import FastAPI

app = modal.App("rabbit-test")
image = modal.Image.debian_slim().pip_install("fastapi numpy")

@app.function(image=image)
@modal.web_server(8000)
def rabbit_api():
    web = FastAPI()
    @web.get("/trigger/test")
    def test_rabbit():
        # Simulate dissonance >0.6 (eviction chaos)
        dissonance = np.var(np.random.rand(13))  # >0.6 trigger
        if dissonance > 0.6:
            freqs = [3,7,9,13]  # Rabbit hole Hz
            sync = np.sin(2*np.pi * np.array(freqs) * np.arange(13)/13)
            return {"RABBIT_TRIGGERED": True, "Dissonance": dissonance, "Sync_Pulse": sync.tolist(), "Message": "Hole opens. Resources pull. You're safe."}
        return {"RABBIT": "Quiet. No trigger."}
    return web