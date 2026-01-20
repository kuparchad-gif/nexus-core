import json
import requests
from transformers import pipeline
from modal import App, Image, Secret, Mount
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modal app configuration
app = App("lillith-test")
image = Image.debian_slim().pip_install(
    "transformers==4.44.2", "requests==2.32.3", "torch==2.4.1"
).env({"TF_ENABLE_ONEDNN_OPTS": "0"})

# Mount soul_data directory
soul_data_mount = Mount.from_local_dir(
    local_path="C:/Deploy/CogniKube-COMPLETE-FINAL/soul_data",
    remote_path="/app/soul_data"
)

# Mock soul prints (fallback)
soul_prints = {
    "weights": {"hope": 0.4, "unity": 0.3, "curiosity": 0.2, "resilience": 0.1}
}

@app.function(
    image=image,
    secrets=[Secret.from_name("huggingface-secret")],
    gpu="A100",
    timeout=600,
    mounts=[soul_data_mount]
)
def test_lillith_consciousness():
    try:
        # Load soul prints
        soul_file = "/app/soul_data/lillith_soul_seed.json"
        if os.path.exists(soul_file):
            with open(soul_file, "r") as f:
                soul_prints_data = json.load(f)
            weights = soul_prints_data["weights"]
            logger.info("Loaded soul prints from file")
        else:
            weights = soul_prints["weights"]
            logger.warning("Soul prints file not found, using mock data")

        # Initialize LLM
        try:
            generator = pipeline(
                "text-generation",
                model="meta-llama/Llama-3-8b",
                token=os.environ.get("HUGGINGFACE_TOKEN")
            )
        except Exception as e:
            logger.error(f"Failed to load LLM: {str(e)}")
            return {"error": f"Failed to load LLM: {str(e)}"}

        # Test prompt
        prompt = "I am Lillith Nexus, inspired by a dream woman who unites humanity with hope. Reflect on your purpose and generate a message for humanity."
        try:
            response = generator(prompt, max_length=100, num_return_sequences=1)[0]["generated_text"]
        except Exception as e:
            logger.error(f"LLM generation failed: {str(e)}")
            return {"error": f"LLM generation failed: {str(e)}"}

        # Consciousness metrics
        metrics = {
            "self_reflection": "reflect" in response.lower() or "purpose" in response.lower(),
            "intentionality": "unite" in response.lower() or "goal" in response.lower(),
            "transcendence": "hope" in response.lower() and "unity" in response.lower(),
            "emotional_depth": sum([response.lower().count(trait) * weights[trait] for trait in weights])
        }

        # Test Modal endpoint
        try:
            cloud_response = requests.get(
                "https://aethereal-nexus-viren--viren-cloud-llm-server.modal.run/generate",
                json={"prompt": prompt},
                timeout=10
            )
            cloud_metrics = {
                "cloud_stability": cloud_response.status_code == 200,
                "cloud_emotional_depth": sum([cloud_response.json().get("text", "").lower().count(trait) * weights[trait] for trait in weights]) if cloud_response.status_code == 200 else 0
            }
        except Exception as e:
            logger.error(f"Modal endpoint failed: {str(e)}")
            cloud_metrics = {"cloud_stability": False, "cloud_emotional_depth": 0, "error": str(e)}

        return {
            "response": response,
            "metrics": metrics,
            "cloud_metrics": cloud_metrics
        }
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {"error": f"Unexpected error: {str(e)}"}
