import os
import asyncio
from agent_framework import FunctionCallContent
from agent_framework.openai import OpenAIChatClient
from openai import AsyncOpenAI

class GitHubModelAgent:
    def __init__(self, model_id: str = "deepseek/DeepSeek-R1", instructions: str = "You are a helpful AI assistant.", tools: list = None):
        self.token = os.getenv("GITHUB_TOKEN")
        if not self.token:
            print("Warning: GITHUB_TOKEN environment variable is not set.")
        
        self.client = AsyncOpenAI(
            base_url="https://models.github.ai/inference",
            api_key=self.token,
        )
        self.model_id = model_id
        self.instructions = instructions
        self.tools = tools

    async def generate_response(self, user_input: str) -> str:
        """Generates a response using the GitHub Model agent."""
        if not self.token:
            return "Error: GITHUB_TOKEN not configured."

        response_text = ""
        async with OpenAIChatClient(
            async_client=self.client,
            model_id=self.model_id
        ).create_agent(
            instructions=self.instructions,
            tools=self.tools,
        ) as agent:
            async for chunk in agent.run_stream([user_input]):
                if chunk.text:
                    response_text += chunk.text
        
        return response_text