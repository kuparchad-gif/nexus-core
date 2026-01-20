# src/agents/production_agent.py  
client = OpenAI(
    base_url="http://your-vllm-server:8000/v1",
    api_key="your-api-key"  # if using auth
)

response = client.chat.completions.create(
    model="llama-2-7b-chat",
    messages=[{"role": "user", "content": "Analyze this portfolio..."}]
)