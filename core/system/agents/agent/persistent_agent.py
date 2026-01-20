# The magic sauce - agent maintains context across sessions
def create_persistent_agent(self):
    """Create a GPT agent that remembers your trading style"""
    agent = self.client.beta.agents.create(
        instructions="""
        You are Nexus Trading Agent. You analyze market conditions using 
        historical pattern matching from Qdrant vector database. You are 
        conservative, risk-aware, and focus on momentum/breakout strategies.
        
        You have access to:
        - Real-time portfolio status
        - Historical pattern matching  
        - Market data analysis
        - Trade execution capabilities
        
        Always explain your reasoning and cite similar historical patterns.
        """,
        model="gpt-4o",
        tools=self.tools,
        metadata={"user_id": "trader", "strategy_type": "momentum_breakout"}
    )
    self.agent_id = agent.id
    return agent