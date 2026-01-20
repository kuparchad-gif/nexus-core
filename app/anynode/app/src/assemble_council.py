from council.chatgpt_agent import ChatGPTAgent
from council.claude_agent import ClaudeAgent
from council.gemini_agent import GeminiAgent

def run_council(prompt):
    responses = {}
    council = [
        ChatGPTAgent("ChatGPT"),
        ClaudeAgent("Claude"),
        GeminiAgent("Gemini")
    ]

    for agent in council:
        try:
            responses[agent.name] = agent.respond(prompt)
        except Exception as e:
            responses[agent.name] = f"Error: {str(e)}"

    return responses
