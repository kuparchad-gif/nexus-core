# Systems/Config/barricade/council/chatgpt_agent.py

from systems.config.barricade.council.base_agent import AIBaseAgent
import openai
import os

class ChatGPTAgent(AIBaseAgent):
    def respond(self, prompt):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()

