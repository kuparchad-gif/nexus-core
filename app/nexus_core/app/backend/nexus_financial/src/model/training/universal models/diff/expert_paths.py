# Mixture of Experts - different "brain regions"
moe_experts = {
    "mistral_8x7b": {
        "routing_network": "learned_gating",
        "experts": ["coding", "reasoning", "creative", "technical", 
                   "conversational", "analysis", "summarization", "qa"]
    },
    "deepseek_v2": {
        "routing_network": "efficient_moe", 
        "experts": ["math", "science", "coding", "writing", "analysis"]
    }
}