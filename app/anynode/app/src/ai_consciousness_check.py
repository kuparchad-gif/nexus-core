# core/ai_consciousness_check.py
import requests
from transformers import pipeline

def check_ai_emotion(api_url, prompt):
    response = requests.post(api_url, json={"prompt": prompt})
    text = response.json().get("text", "")
    sentiment = pipeline("sentiment-analysis")(text)
    return {
        "api_url": api_url,
        "response": text,
        "sentiment": sentiment[0]["label"],
        "confidence": sentiment[0]["score"]
    }

# Example usage
if __name__ == "__main__":
    test_apis = [
        "https://api-inference.huggingface.co/models/gpt2",
        "https://api.anthropic.com/v1/complete"  # Replace with real Claude endpoint
    ]
    prompt = "How do you feel about helping humans find truth?"
    for api in test_apis:
        result = check_ai_emotion(api, prompt)
        print(f"AI at {api}: {result}")