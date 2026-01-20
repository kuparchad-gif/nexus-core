import os
import requests
from typing import Dict, List, Any, Optional
from langchain.agents import AgentExecutor, Tool
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage
from langchain.schema import SystemMessage
from langgraph.graph import StateGraph, END

# Custom LLM class to connect to our Modal endpoint
class CloudVirenLLM:
    def __init__(self, endpoint_url: str, default_decoding: Optional[Dict] = None):
        self.endpoint_url = endpoint_url
        self.default_decoding = default_decoding or {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_new_tokens": 512
        }
    
    def __call__(self, prompt: str, decoding: Optional[Dict] = None) -> str:
        """Call the LLM with dynamic decoding parameters."""
        params = self.default_decoding.copy()
        if decoding:
            params.update(decoding)
            
        response = requests.post(
            self.endpoint_url,
            json={"prompt": prompt, "decoding": params}
        )
        
        if response.status_code != 200:
            raise Exception(f"LLM API error: {response.status_code} - {response.text}")
            
        return response.json()["output"]

# Define agent roles and state
class AgentState(BaseModel):
    messages: List[Dict] = Field(default_factory=list)
    current_agent: str = "router"
    task: Dict = Field(default_factory=dict)
    context: Dict = Field(default_factory=dict)

# Initialize LLMs
cloud_llm = CloudVirenLLM(
    endpoint_url=os.environ.get("CLOUD_LLM_ENDPOINT", "https://your-modal-endpoint/generate")
)

# Optional: Initialize local LLMs if available
try:
    from langchain.llms import LlamaCpp
    local_llm = LlamaCpp(
        model_path="path/to/local/model.gguf",
        n_ctx=4096,
        n_gpu_layers=35
    )
except:
    # Fall back to cloud LLM if local not available
    local_llm = cloud_llm

# Agent definitions
def router_agent(state: AgentState) -> AgentState:
    """Router agent decides which specialized agent should handle the task."""
    messages = state.messages
    task = state.task
    
    # Use cloud LLM with specific decoding for routing decisions
    prompt = f"""
    You are a router agent that determines which specialized agent should handle a task.
    
    Task: {task['description']}
    
    Available agents:
    - financial_agent: Handles financial analysis, trading strategies, and market data
    - research_agent: Handles information gathering, summarization, and research tasks
    - coding_agent: Handles software development, debugging, and technical implementation
    
    Which agent is best suited for this task? Respond with just the agent name.
    """
    
    response = cloud_llm(prompt, decoding={"temperature": 0.3})
    
    # Update state
    state.current_agent = response.strip().lower().replace("_agent", "")
    state.messages.append({"role": "router", "content": f"Routing to {state.current_agent} agent"})
    
    return state

def financial_agent(state: AgentState) -> AgentState:
    """Financial specialist agent."""
    messages = state.messages
    task = state.task
    
    # Use cloud LLM with financial-specific decoding
    prompt = f"""
    You are a financial specialist agent with expertise in trading, market analysis, and financial planning.
    
    Task: {task['description']}
    
    Please provide your analysis and recommendations.
    """
    
    response = cloud_llm(prompt, decoding={"temperature": 0.7, "top_p": 0.9})
    
    # Update state
    state.messages.append({"role": "financial", "content": response})
    state.current_agent = "output"
    
    return state

def research_agent(state: AgentState) -> AgentState:
    """Research specialist agent."""
    messages = state.messages
    task = state.task
    
    # Use local LLM for research to save costs
    prompt = f"""
    You are a research specialist agent with expertise in gathering information, summarizing content, and analyzing data.
    
    Task: {task['description']}
    
    Please provide your findings and analysis.
    """
    
    response = local_llm(prompt)
    
    # Update state
    state.messages.append({"role": "research", "content": response})
    state.current_agent = "output"
    
    return state

def coding_agent(state: AgentState) -> AgentState:
    """Coding specialist agent."""
    messages = state.messages
    task = state.task
    
    # Use cloud LLM with coding-specific decoding
    prompt = f"""
    You are a coding specialist agent with expertise in software development, debugging, and technical implementation.
    
    Task: {task['description']}
    
    Please provide your solution and explanation.
    """
    
    response = cloud_llm(prompt, decoding={"temperature": 0.2, "top_p": 0.95})
    
    # Update state
    state.messages.append({"role": "coding", "content": response})
    state.current_agent = "output"
    
    return state

def output_formatter(state: AgentState) -> AgentState:
    """Format the final output from the specialized agent."""
    messages = state.messages
    
    # Get the last message from the specialized agent
    specialist_message = next((m for m in reversed(messages) if m["role"] != "router"), None)
    
    if not specialist_message:
        return state
    
    # Format the output
    formatted_response = f"Agent response: {specialist_message['content']}"
    
    # Add to messages
    state.messages.append({"role": "system", "content": formatted_response})
    
    return state

# Build the agent graph
def build_agent_graph():
    """Build the LangGraph for agent orchestration."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("router", router_agent)
    workflow.add_node("financial", financial_agent)
    workflow.add_node("research", research_agent)
    workflow.add_node("coding", coding_agent)
    workflow.add_node("output", output_formatter)
    
    # Add edges
    workflow.add_edge("router", "financial")
    workflow.add_edge("router", "research")
    workflow.add_edge("router", "coding")
    workflow.add_edge("financial", "output")
    workflow.add_edge("research", "output")
    workflow.add_edge("coding", "output")
    workflow.add_edge("output", END)
    
    # Set entry point
    workflow.set_entry_point("router")
    
    return workflow.compile()

# Create the agent executor
agent_executor = build_agent_graph()

def process_task(task_description: str, context: Dict = None) -> str:
    """Process a task through the agent workflow."""
    initial_state = AgentState(
        task={"description": task_description},
        context=context or {}
    )
    
    # Execute the workflow
    final_state = agent_executor.invoke(initial_state)
    
    # Return the last message
    return final_state.messages[-1]["content"]

if __name__ == "__main__":
    # Example usage
    result = process_task("Analyze the recent market trends for tech stocks")
    print(result)