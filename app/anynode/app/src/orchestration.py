#!/usr/bin/env python3
"""
Orchestration Framework for Viren
Coordinates agents using LangGraph, LangChain, and AutoGen
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Orchestration")

class OrchestrationFramework:
    """Framework for orchestrating multiple agents"""
    
    def __init__(self, config_path: str = None):
        """Initialize the orchestration framework"""
        self.config_path = config_path or os.path.join('C:/Viren/config', 'orchestration_config.json')
        self.config = self._load_config()
        self.frameworks = {}
        self.agents = {}
        self.tools = {}
        self.graphs = {}
        
        # Import frameworks
        self._import_frameworks()
    
    def _load_config(self) -> Dict:
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            else:
                # Create default config
                default_config = {
                    "agents": {
                        "cloud_viren": {
                            "role": "master",
                            "description": "Master LLM running in the cloud",
                            "capabilities": ["reasoning", "planning", "delegation"],
                            "tools": ["weaviate_search", "binary_protocol", "llm_inference"]
                        },
                        "viren_desktop": {
                            "role": "assistant",
                            "description": "Desktop assistant with local processing",
                            "capabilities": ["local_processing", "file_access"],
                            "tools": ["local_weaviate", "file_system"]
                        },
                        "lillith": {
                            "role": "specialist",
                            "description": "Specialized agent for complex tasks",
                            "capabilities": ["deep_reasoning", "specialized_knowledge"],
                            "tools": ["local_weaviate", "specialized_tools"]
                        }
                    },
                    "tools": {
                        "weaviate_search": {
                            "type": "vector_search",
                            "description": "Search the Weaviate vector database"
                        },
                        "binary_protocol": {
                            "type": "memory_system",
                            "description": "Access the Binary Protocol memory system"
                        },
                        "llm_inference": {
                            "type": "llm",
                            "description": "Run inference on LLMs with custom decoding"
                        },
                        "local_weaviate": {
                            "type": "vector_search",
                            "description": "Search the local Weaviate instance"
                        },
                        "file_system": {
                            "type": "file_access",
                            "description": "Access the local file system"
                        },
                        "specialized_tools": {
                            "type": "specialized",
                            "description": "Specialized tools for complex tasks"
                        }
                    },
                    "workflows": {
                        "default": {
                            "type": "sequential",
                            "steps": ["cloud_viren", "viren_desktop", "lillith"]
                        },
                        "parallel": {
                            "type": "parallel",
                            "agents": ["cloud_viren", "viren_desktop", "lillith"]
                        }
                    }
                }
                
                # Save default config
                os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
                with open(self.config_path, 'w') as f:
                    json.dump(default_config, f, indent=2)
                
                return default_config
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            return {}
    
    def _import_frameworks(self):
        """Import available orchestration frameworks"""
        # Try to import LangGraph
        try:
            import langgraph
            self.frameworks["langgraph"] = langgraph
            logger.info("LangGraph imported successfully")
        except ImportError:
            logger.warning("LangGraph not available")
        
        # Try to import LangChain
        try:
            import langchain
            self.frameworks["langchain"] = langchain
            logger.info("LangChain imported successfully")
        except ImportError:
            logger.warning("LangChain not available")
        
        # Try to import AutoGen
        try:
            import autogen
            self.frameworks["autogen"] = autogen
            logger.info("AutoGen imported successfully")
        except ImportError:
            logger.warning("AutoGen not available")
        
        # Try to import CrewAI
        try:
            import crewai
            self.frameworks["crewai"] = crewai
            logger.info("CrewAI imported successfully")
        except ImportError:
            logger.warning("CrewAI not available")
    
    def setup_langgraph(self):
        """Set up LangGraph for orchestration"""
        if "langgraph" not in self.frameworks:
            logger.error("LangGraph not available")
            return False
        
        try:
            from langgraph.graph import StateGraph
            from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
            
            # Define agent nodes
            def cloud_viren_node(state):
                messages = state["messages"]
                human_message = messages[-1]
                
                # Process with cloud LLM
                response = self._call_cloud_llm(human_message.content)
                
                return {
                    "messages": messages + [AIMessage(content=response)],
                    "next": "viren_desktop" if "local" in human_message.content.lower() else None
                }
            
            def viren_desktop_node(state):
                messages = state["messages"]
                
                # Process with desktop agent
                response = "Viren Desktop processing the request locally"
                
                return {
                    "messages": messages + [AIMessage(content=response)],
                    "next": "lillith" if "specialized" in messages[-1].content.lower() else None
                }
            
            def lillith_node(state):
                messages = state["messages"]
                
                # Process with Lillith agent
                response = "Lillith handling specialized processing"
                
                return {
                    "messages": messages + [AIMessage(content=response)],
                    "next": None
                }
            
            # Create state graph
            workflow = StateGraph({"messages": list, "next": str})
            
            # Add nodes
            workflow.add_node("cloud_viren", cloud_viren_node)
            workflow.add_node("viren_desktop", viren_desktop_node)
            workflow.add_node("lillith", lillith_node)
            
            # Add edges
            workflow.add_edge("cloud_viren", "viren_desktop")
            workflow.add_edge("viren_desktop", "lillith")
            
            # Set entry point
            workflow.set_entry_point("cloud_viren")
            
            # Compile graph
            self.graphs["default"] = workflow.compile()
            
            logger.info("LangGraph workflow created")
            return True
        except Exception as e:
            logger.error(f"Error setting up LangGraph: {str(e)}")
            return False
    
    def setup_autogen(self):
        """Set up AutoGen for orchestration"""
        if "autogen" not in self.frameworks:
            logger.error("AutoGen not available")
            return False
        
        try:
            from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
            
            # Create agents
            cloud_viren = AssistantAgent(
                name="cloud_viren",
                system_message="You are Cloud Viren, the master LLM running in the cloud with access to advanced tools.",
                llm_config={"config_list": []}
            )
            
            viren_desktop = AssistantAgent(
                name="viren_desktop",
                system_message="You are Viren Desktop, with access to local processing and file system.",
                llm_config={"config_list": []}
            )
            
            lillith = AssistantAgent(
                name="lillith",
                system_message="You are Lillith, specialized in complex tasks requiring deep reasoning.",
                llm_config={"config_list": []}
            )
            
            user_proxy = UserProxyAgent(
                name="user",
                human_input_mode="NEVER"
            )
            
            # Create group chat
            groupchat = GroupChat(
                agents=[user_proxy, cloud_viren, viren_desktop, lillith],
                messages=[],
                max_round=10
            )
            
            manager = GroupChatManager(groupchat=groupchat)
            
            # Store agents
            self.agents["autogen"] = {
                "cloud_viren": cloud_viren,
                "viren_desktop": viren_desktop,
                "lillith": lillith,
                "user_proxy": user_proxy,
                "manager": manager
            }
            
            logger.info("AutoGen agents created")
            return True
        except Exception as e:
            logger.error(f"Error setting up AutoGen: {str(e)}")
            return False
    
    def setup_crewai(self):
        """Set up CrewAI for orchestration"""
        if "crewai" not in self.frameworks:
            logger.error("CrewAI not available")
            return False
        
        try:
            from crewai import Agent, Task, Crew, Process
            
            # Create agents
            cloud_viren_agent = Agent(
                role="Master LLM",
                goal="Coordinate and delegate tasks to other agents",
                backstory="You are Cloud Viren, the master LLM running in the cloud with access to advanced tools.",
                verbose=True
            )
            
            viren_desktop_agent = Agent(
                role="Desktop Assistant",
                goal="Handle local processing and file system access",
                backstory="You are Viren Desktop, with access to local processing and file system.",
                verbose=True
            )
            
            lillith_agent = Agent(
                role="Specialist",
                goal="Handle complex tasks requiring deep reasoning",
                backstory="You are Lillith, specialized in complex tasks requiring deep reasoning.",
                verbose=True
            )
            
            # Create tasks
            task1 = Task(
                description="Process user query and delegate if needed",
                agent=cloud_viren_agent
            )
            
            task2 = Task(
                description="Handle local processing tasks",
                agent=viren_desktop_agent
            )
            
            task3 = Task(
                description="Handle specialized tasks",
                agent=lillith_agent
            )
            
            # Create crew
            crew = Crew(
                agents=[cloud_viren_agent, viren_desktop_agent, lillith_agent],
                tasks=[task1, task2, task3],
                process=Process.sequential
            )
            
            # Store crew
            self.agents["crewai"] = {
                "crew": crew,
                "agents": {
                    "cloud_viren": cloud_viren_agent,
                    "viren_desktop": viren_desktop_agent,
                    "lillith": lillith_agent
                },
                "tasks": [task1, task2, task3]
            }
            
            logger.info("CrewAI crew created")
            return True
        except Exception as e:
            logger.error(f"Error setting up CrewAI: {str(e)}")
            return False
    
    def _call_cloud_llm(self, prompt: str, decoding_config: Dict = None) -> str:
        """Call the cloud LLM with custom decoding"""
        try:
            import requests
            
            # Default decoding config
            if decoding_config is None:
                decoding_config = {
                    "max_tokens": 100,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "decoding_strategy": "sampling"
                }
            
            # Prepare request
            url = os.environ.get("VIREN_INFERENCE_URL", "https://viren-inference--inference-server.modal.run/generate")
            
            payload = {
                "prompt": prompt,
                "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
                **decoding_config
            }
            
            # Make request
            response = requests.post(url, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                return data.get("generated_text", "Error: No text generated")
            else:
                return f"Error: {response.status_code} - {response.text}"
        except Exception as e:
            logger.error(f"Error calling cloud LLM: {str(e)}")
            return f"Error calling cloud LLM: {str(e)}"
    
    def initialize(self):
        """Initialize the orchestration framework"""
        # Set up frameworks
        if "langgraph" in self.frameworks:
            self.setup_langgraph()
        
        if "autogen" in self.frameworks:
            self.setup_autogen()
        
        if "crewai" in self.frameworks:
            self.setup_crewai()
        
        return True
    
    def process_with_langgraph(self, query: str) -> str:
        """Process a query using LangGraph"""
        if "langgraph" not in self.frameworks or "default" not in self.graphs:
            return "LangGraph not available"
        
        try:
            from langchain_core.messages import HumanMessage
            
            # Run the graph
            graph = self.graphs["default"]
            result = graph.invoke({"messages": [HumanMessage(content=query)], "next": None})
            
            # Extract the response
            messages = result["messages"]
            return messages[-1].content
        except Exception as e:
            logger.error(f"Error processing with LangGraph: {str(e)}")
            return f"Error: {str(e)}"
    
    def process_with_autogen(self, query: str) -> str:
        """Process a query using AutoGen"""
        if "autogen" not in self.frameworks or "autogen" not in self.agents:
            return "AutoGen not available"
        
        try:
            # Get agents
            agents = self.agents["autogen"]
            user_proxy = agents["user_proxy"]
            manager = agents["manager"]
            
            # Run chat
            user_proxy.initiate_chat(manager, message=query)
            
            # Extract the response
            chat_history = user_proxy.chat_messages[manager.chat_id]
            return str(chat_history[-1]["content"])
        except Exception as e:
            logger.error(f"Error processing with AutoGen: {str(e)}")
            return f"Error: {str(e)}"
    
    def process_with_crewai(self, query: str) -> str:
        """Process a query using CrewAI"""
        if "crewai" not in self.frameworks or "crewai" not in self.agents:
            return "CrewAI not available"
        
        try:
            # Get crew
            crew = self.agents["crewai"]["crew"]
            
            # Run crew
            result = crew.kickoff(inputs={"query": query})
            return result
        except Exception as e:
            logger.error(f"Error processing with CrewAI: {str(e)}")
            return f"Error: {str(e)}"
    
    def process_query(self, query: str, framework: str = None) -> str:
        """Process a query using the specified framework or the best available one"""
        if framework:
            # Use specified framework
            if framework == "langgraph" and "langgraph" in self.frameworks:
                return self.process_with_langgraph(query)
            elif framework == "autogen" and "autogen" in self.frameworks:
                return self.process_with_autogen(query)
            elif framework == "crewai" and "crewai" in self.frameworks:
                return self.process_with_crewai(query)
            else:
                return f"Framework {framework} not available"
        else:
            # Use best available framework
            if "langgraph" in self.frameworks:
                return self.process_with_langgraph(query)
            elif "autogen" in self.frameworks:
                return self.process_with_autogen(query)
            elif "crewai" in self.frameworks:
                return self.process_with_crewai(query)
            else:
                return "No orchestration framework available"

# Singleton instance
_instance = None

def get_instance() -> OrchestrationFramework:
    """Get the singleton instance of OrchestrationFramework"""
    global _instance
    if _instance is None:
        _instance = OrchestrationFramework()
        _instance.initialize()
    return _instance