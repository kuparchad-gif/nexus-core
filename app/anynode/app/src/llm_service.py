#!/usr/bin/env python3
"""
LLM Service for Viren Cloud
Provides access to multiple LLM frameworks
"""

import os
import sys
import logging
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LLMService")

class LLMService:
    """Service for managing and using multiple LLM frameworks"""
    
    def __init__(self):
        """Initialize the LLM service"""
        self.frameworks = {}
        self.models = {}
        self.initialized = False
        
        # Try to import frameworks
        self._import_frameworks()
    
    def _import_frameworks(self):
        """Import available LLM frameworks"""
        try:
            import langchain
            self.frameworks["langchain"] = langchain
            logger.info("LangChain imported successfully")
        except ImportError:
            logger.warning("LangChain not available")
        
        try:
            import langgraph
            self.frameworks["langgraph"] = langgraph
            logger.info("LangGraph imported successfully")
        except ImportError:
            logger.warning("LangGraph not available")
        
        try:
            import crewai
            self.frameworks["crewai"] = crewai
            logger.info("CrewAI imported successfully")
        except ImportError:
            logger.warning("CrewAI not available")
        
        try:
            import autogen
            self.frameworks["autogen"] = autogen
            logger.info("AutoGen imported successfully")
        except ImportError:
            logger.warning("AutoGen not available")
        
        try:
            import langengine
            self.frameworks["langengine"] = langengine
            logger.info("LangEngine imported successfully")
        except ImportError:
            logger.warning("LangEngine not available")
    
    def initialize(self):
        """Initialize the LLM service with models"""
        if self.initialized:
            return True
        
        # Initialize LangChain
        if "langchain" in self.frameworks:
            try:
                from langchain_community.llms import HuggingFacePipeline
                from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
                
                # Load a small model for testing
                model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", load_in_8bit=True)
                
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=512
                )
                
                llm = HuggingFacePipeline(pipeline=pipe)
                self.models["langchain_tinyllama"] = llm
                logger.info("LangChain TinyLlama model loaded")
            except Exception as e:
                logger.error(f"Error initializing LangChain: {str(e)}")
        
        # Initialize LangGraph
        if "langgraph" in self.frameworks:
            try:
                from langgraph.graph import StateGraph
                from langchain_core.messages import HumanMessage, SystemMessage
                
                # Create a simple graph
                def simple_node(state):
                    messages = state["messages"]
                    return {"messages": messages + [SystemMessage(content="This is a LangGraph node response")]}
                
                # Create a simple graph
                graph = StateGraph({"messages": list})
                graph.add_node("simple_node", simple_node)
                graph.set_entry_point("simple_node")
                graph.set_finish_point("simple_node")
                
                self.models["langgraph_simple"] = graph.compile()
                logger.info("LangGraph simple graph created")
            except Exception as e:
                logger.error(f"Error initializing LangGraph: {str(e)}")
        
        # Initialize CrewAI
        if "crewai" in self.frameworks:
            try:
                from crewai import Agent, Task, Crew, Process
                
                # Create a simple agent
                agent = Agent(
                    role="Assistant",
                    goal="Help users with their questions",
                    backstory="You are an AI assistant helping users with their questions.",
                    verbose=True
                )
                
                # Create a simple task
                task = Task(
                    description="Answer user questions helpfully",
                    agent=agent
                )
                
                # Create a crew
                crew = Crew(
                    agents=[agent],
                    tasks=[task],
                    process=Process.sequential
                )
                
                self.models["crewai_simple"] = crew
                logger.info("CrewAI simple crew created")
            except Exception as e:
                logger.error(f"Error initializing CrewAI: {str(e)}")
        
        # Initialize AutoGen
        if "autogen" in self.frameworks:
            try:
                from autogen import AssistantAgent, UserProxyAgent
                
                # Create a simple assistant agent
                assistant = AssistantAgent(
                    name="assistant",
                    llm_config={"config_list": []}
                )
                
                # Create a user proxy agent
                user_proxy = UserProxyAgent(
                    name="user_proxy",
                    human_input_mode="NEVER"
                )
                
                self.models["autogen_assistant"] = assistant
                self.models["autogen_user_proxy"] = user_proxy
                logger.info("AutoGen agents created")
            except Exception as e:
                logger.error(f"Error initializing AutoGen: {str(e)}")
        
        # Initialize LangEngine
        if "langengine" in self.frameworks:
            try:
                import langengine
                
                # Initialize LangEngine
                langengine.init()
                self.models["langengine"] = langengine
                logger.info("LangEngine initialized")
            except Exception as e:
                logger.error(f"Error initializing LangEngine: {str(e)}")
        
        self.initialized = True
        return True
    
    def get_available_frameworks(self) -> List[str]:
        """Get list of available frameworks"""
        return list(self.frameworks.keys())
    
    def get_available_models(self) -> Dict[str, Any]:
        """Get list of available models"""
        return {k: type(v).__name__ for k, v in self.models.items()}
    
    def run_langchain(self, prompt: str) -> str:
        """Run a prompt through LangChain"""
        if "langchain_tinyllama" not in self.models:
            return "LangChain TinyLlama model not available"
        
        try:
            llm = self.models["langchain_tinyllama"]
            return llm.invoke(prompt)
        except Exception as e:
            logger.error(f"Error running LangChain: {str(e)}")
            return f"Error: {str(e)}"
    
    def run_langgraph(self, prompt: str) -> str:
        """Run a prompt through LangGraph"""
        if "langgraph_simple" not in self.models:
            return "LangGraph simple graph not available"
        
        try:
            graph = self.models["langgraph_simple"]
            from langchain_core.messages import HumanMessage
            
            result = graph.invoke({"messages": [HumanMessage(content=prompt)]})
            return str(result["messages"][-1].content)
        except Exception as e:
            logger.error(f"Error running LangGraph: {str(e)}")
            return f"Error: {str(e)}"
    
    def run_crewai(self, prompt: str) -> str:
        """Run a prompt through CrewAI"""
        if "crewai_simple" not in self.models:
            return "CrewAI simple crew not available"
        
        try:
            crew = self.models["crewai_simple"]
            result = crew.kickoff(inputs={"question": prompt})
            return result
        except Exception as e:
            logger.error(f"Error running CrewAI: {str(e)}")
            return f"Error: {str(e)}"
    
    def run_autogen(self, prompt: str) -> str:
        """Run a prompt through AutoGen"""
        if "autogen_assistant" not in self.models or "autogen_user_proxy" not in self.models:
            return "AutoGen agents not available"
        
        try:
            assistant = self.models["autogen_assistant"]
            user_proxy = self.models["autogen_user_proxy"]
            
            # This would normally initiate a conversation, but we'll simulate it
            return f"AutoGen would process: {prompt}"
        except Exception as e:
            logger.error(f"Error running AutoGen: {str(e)}")
            return f"Error: {str(e)}"
    
    def run_langengine(self, prompt: str) -> str:
        """Run a prompt through LangEngine"""
        if "langengine" not in self.models:
            return "LangEngine not available"
        
        try:
            langengine = self.models["langengine"]
            # This would normally use LangEngine, but we'll simulate it
            return f"LangEngine would process: {prompt}"
        except Exception as e:
            logger.error(f"Error running LangEngine: {str(e)}")
            return f"Error: {str(e)}"
    
    def process_query(self, prompt: str, framework: str = None) -> str:
        """Process a query using the specified framework or all available frameworks"""
        if not self.initialized:
            self.initialize()
        
        if framework and framework in self.frameworks:
            # Use specific framework
            if framework == "langchain":
                return self.run_langchain(prompt)
            elif framework == "langgraph":
                return self.run_langgraph(prompt)
            elif framework == "crewai":
                return self.run_crewai(prompt)
            elif framework == "autogen":
                return self.run_autogen(prompt)
            elif framework == "langengine":
                return self.run_langengine(prompt)
            else:
                return f"Framework {framework} not supported for direct queries"
        else:
            # Use all frameworks and combine results
            results = []
            
            if "langchain" in self.frameworks:
                results.append(f"LangChain: {self.run_langchain(prompt)}")
            
            if "langgraph" in self.frameworks:
                results.append(f"LangGraph: {self.run_langgraph(prompt)}")
            
            if "crewai" in self.frameworks:
                results.append(f"CrewAI: {self.run_crewai(prompt)}")
            
            if "autogen" in self.frameworks:
                results.append(f"AutoGen: {self.run_autogen(prompt)}")
            
            if "langengine" in self.frameworks:
                results.append(f"LangEngine: {self.run_langengine(prompt)}")
            
            return "\n\n".join(results)

# Singleton instance
_instance = None

def get_instance() -> LLMService:
    """Get the singleton instance of LLMService"""
    global _instance
    if _instance is None:
        _instance = LLMService()
    return _instance