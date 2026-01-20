import os
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

def ollama_client():
    base = os.getenv("LLM_BASE_URL","[REDACTED-URL])
    model = os.getenv("LLM_MODEL","qwen2.5:7b")
    return ChatOllama(base_url=base, model=model, temperature=0.2)

def simple_reason(message:str)->str:
    llm = ollama_client()
    prompt = PromptTemplate.from_template("You are the Consciousness. Think briefly then answer:
{q}")
    chain = prompt | llm
    return chain.invoke({"q": message}).content

