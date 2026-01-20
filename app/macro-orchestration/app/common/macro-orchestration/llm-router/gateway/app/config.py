import os

GATEWAY_HOST  =  os.getenv("GATEWAY_HOST", "0.0.0.0")
GATEWAY_PORT  =  int(os.getenv("GATEWAY_PORT", "8000"))
MODE_DEFAULT  =  os.getenv("MODE_DEFAULT", "top_k")
TOP_K_DEFAULT  =  int(os.getenv("TOP_K_DEFAULT", "2"))
COMBINER_DEFAULT  =  os.getenv("COMBINER_DEFAULT", "concat")

EXPERTS  =  {
    "math": os.getenv("EXPERT_MATH_URL", "http://math:7001"),
    "graph": os.getenv("EXPERT_GRAPH_URL", "http://graph:7002"),
    "sentiment": os.getenv("EXPERT_SENTIMENT_URL", "http://sentiment:7003"),
    "code": os.getenv("EXPERT_CODE_URL", "http://code:7004"),
    "llm": os.getenv("EXPERT_LLM_URL", "http://llm-expert:7005"),
}
LM_TOOLBOX_URL  =  os.getenv("LM_TOOLBOX_URL", "http://host.docker.internal:8089")
GATING_LM = os.getenv("GATING_LM", "false").lower() == "true"

PROMPT_DIR  =  "/data/prompts"
os.makedirs(PROMPT_DIR, exist_ok = True)
