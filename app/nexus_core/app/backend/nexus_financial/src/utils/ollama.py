# src/agents/local_agent.py
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # not needed but required
)

response = client.chat.completions.create(
    model="llama2",
    messages=[{"role": "user", "content": "Analyze this portfolio..."}]
)