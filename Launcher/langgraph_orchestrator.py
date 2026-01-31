"""
LangGraph Orchestration Layer
Provides stateful agent workflows and multi-agent coordination
"""

from typing import Dict, Any, List, Optional, Callable, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import operator
import logging
from enum import Enum

logger = logging.getLogger(__name__)


# ==================== STATE DEFINITIONS ====================

class AgentState(TypedDict):
    """Base state for agent workflows"""
    messages: Annotated[List[BaseMessage], operator.add]
    current_task: str
    context: Dict[str, Any]
    results: Dict[str, Any]
    next_action: str
    iteration: int
    max_iterations: int


class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


# ==================== LANGGRAPH ORCHESTRATOR ====================

class LangGraphOrchestrator:
    """
    LangGraph-based workflow orchestration
    - Stateful agent workflows
    - Multi-agent coordination
    - Conditional routing
    - Human-in-the-loop
    - Checkpointing and persistence
    """
    
    def __init__(self, model_name: str = "gpt-4.1-mini", temperature: float = 0.7):
        self.model = ChatOpenAI(model=model_name, temperature=temperature)
        self.checkpointer = MemorySaver()
        self.workflows = {}
        
    # ==================== WORKFLOW CREATION ====================
    
    def create_workflow(self, workflow_id: str, 
                       initial_state: Optional[Dict[str, Any]] = None) -> StateGraph:
        """Create a new stateful workflow"""
        try:
            # Initialize state graph
            workflow = StateGraph(AgentState)
            
            # Set default initial state
            if initial_state is None:
                initial_state = {
                    "messages": [],
                    "current_task": "",
                    "context": {},
                    "results": {},
                    "next_action": "start",
                    "iteration": 0,
                    "max_iterations": 10
                }
            
            self.workflows[workflow_id] = {
                "graph": workflow,
                "initial_state": initial_state,
                "status": WorkflowStatus.PENDING
            }
            
            logger.info(f"Workflow created: {workflow_id}")
            return workflow
            
        except Exception as e:
            logger.error(f"Failed to create workflow: {e}")
            raise
    
    def add_node(self, workflow_id: str, node_name: str, 
                 node_func: Callable) -> None:
        """Add a node (processing step) to workflow"""
        try:
            workflow = self.workflows[workflow_id]["graph"]
            workflow.add_node(node_name, node_func)
            logger.info(f"Node added to {workflow_id}: {node_name}")
        except Exception as e:
            logger.error(f"Failed to add node: {e}")
            raise
    
    def add_edge(self, workflow_id: str, from_node: str, to_node: str) -> None:
        """Add a simple edge between nodes"""
        try:
            workflow = self.workflows[workflow_id]["graph"]
            workflow.add_edge(from_node, to_node)
            logger.info(f"Edge added: {from_node} -> {to_node}")
        except Exception as e:
            logger.error(f"Failed to add edge: {e}")
            raise
    
    def add_conditional_edge(self, workflow_id: str, from_node: str,
                            condition_func: Callable,
                            edge_mapping: Dict[str, str]) -> None:
        """
        Add conditional edge with routing logic
        condition_func: Function that returns key for edge_mapping
        edge_mapping: {condition_result: target_node}
        """
        try:
            workflow = self.workflows[workflow_id]["graph"]
            workflow.add_conditional_edges(from_node, condition_func, edge_mapping)
            logger.info(f"Conditional edge added from {from_node}")
        except Exception as e:
            logger.error(f"Failed to add conditional edge: {e}")
            raise
    
    def set_entry_point(self, workflow_id: str, node_name: str) -> None:
        """Set the entry point for workflow"""
        try:
            workflow = self.workflows[workflow_id]["graph"]
            workflow.set_entry_point(node_name)
            logger.info(f"Entry point set: {node_name}")
        except Exception as e:
            logger.error(f"Failed to set entry point: {e}")
            raise
    
    def compile_workflow(self, workflow_id: str) -> Any:
        """Compile workflow for execution"""
        try:
            workflow = self.workflows[workflow_id]["graph"]
            compiled = workflow.compile(checkpointer=self.checkpointer)
            self.workflows[workflow_id]["compiled"] = compiled
            logger.info(f"Workflow compiled: {workflow_id}")
            return compiled
        except Exception as e:
            logger.error(f"Failed to compile workflow: {e}")
            raise
    
    # ==================== WORKFLOW EXECUTION ====================
    
    def execute_workflow(self, workflow_id: str, 
                        input_data: Dict[str, Any],
                        config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute workflow with input data"""
        try:
            if "compiled" not in self.workflows[workflow_id]:
                self.compile_workflow(workflow_id)
            
            compiled = self.workflows[workflow_id]["compiled"]
            self.workflows[workflow_id]["status"] = WorkflowStatus.RUNNING
            
            # Execute
            if config is None:
                config = {"configurable": {"thread_id": "demo_thread"}}
            result = compiled.invoke(input_data, config=config)
            
            self.workflows[workflow_id]["status"] = WorkflowStatus.COMPLETED
            logger.info(f"Workflow executed: {workflow_id}")
            
            return result
            
        except Exception as e:
            self.workflows[workflow_id]["status"] = WorkflowStatus.FAILED
            logger.error(f"Workflow execution failed: {e}")
            raise
    
    def stream_workflow(self, workflow_id: str, 
                       input_data: Dict[str, Any],
                       config: Optional[Dict[str, Any]] = None):
        """Stream workflow execution (yields intermediate states)"""
        try:
            if "compiled" not in self.workflows[workflow_id]:
                self.compile_workflow(workflow_id)
            
            compiled = self.workflows[workflow_id]["compiled"]
            self.workflows[workflow_id]["status"] = WorkflowStatus.RUNNING
            
            if config is None:
                config = {"configurable": {"thread_id": "demo_thread"}}
            for state in compiled.stream(input_data, config=config):
                yield state
            
            self.workflows[workflow_id]["status"] = WorkflowStatus.COMPLETED
            
        except Exception as e:
            self.workflows[workflow_id]["status"] = WorkflowStatus.FAILED
            logger.error(f"Workflow streaming failed: {e}")
            raise
    
    # ==================== PRE-BUILT WORKFLOWS ====================
    
    def create_simple_agent_workflow(self, workflow_id: str) -> str:
        """
        Create a simple agent workflow:
        start -> think -> act -> observe -> decide -> (continue or end)
        """
        workflow = self.create_workflow(workflow_id)
        
        # Define nodes
        def think_node(state: AgentState) -> AgentState:
            """Agent thinks about the task"""
            messages = state["messages"]
            context = state["context"]
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful AI agent. Analyze the task and plan your approach."),
                ("human", "{task}")
            ])
            
            response = self.model.invoke(
                prompt.format_messages(task=state["current_task"])
            )
            
            messages.append(AIMessage(content=response.content))
            state["context"]["plan"] = response.content
            state["next_action"] = "act"
            
            return state
        
        def act_node(state: AgentState) -> AgentState:
            """Agent takes action"""
            plan = state["context"].get("plan", "")
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Based on your plan, decide what action to take."),
                ("human", "Plan: {plan}")
            ])
            
            response = self.model.invoke(
                prompt.format_messages(plan=plan)
            )
            
            state["messages"].append(AIMessage(content=response.content))
            state["context"]["action"] = response.content
            state["next_action"] = "observe"
            
            return state
        
        def observe_node(state: AgentState) -> AgentState:
            """Agent observes results"""
            action = state["context"].get("action", "")
            
            # Simulate observation
            state["context"]["observation"] = f"Completed action: {action}"
            state["next_action"] = "decide"
            state["iteration"] += 1
            
            return state
        
        def decide_node(state: AgentState) -> AgentState:
            """Agent decides next step"""
            observation = state["context"].get("observation", "")
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Based on the observation, decide if the task is complete or if you need to continue."),
                ("human", "Observation: {observation}")
            ])
            
            response = self.model.invoke(
                prompt.format_messages(observation=observation)
            )
            
            state["messages"].append(AIMessage(content=response.content))
            
            # Check if complete or max iterations reached
            if "complete" in response.content.lower() or state["iteration"] >= state["max_iterations"]:
                state["next_action"] = "end"
            else:
                state["next_action"] = "think"
            
            return state
        
        def should_continue(state: AgentState) -> str:
            """Routing function"""
            return state["next_action"]
        
        # Add nodes
        self.add_node(workflow_id, "think", think_node)
        self.add_node(workflow_id, "act", act_node)
        self.add_node(workflow_id, "observe", observe_node)
        self.add_node(workflow_id, "decide", decide_node)
        
        # Add edges
        self.add_edge(workflow_id, "think", "act")
        self.add_edge(workflow_id, "act", "observe")
        self.add_edge(workflow_id, "observe", "decide")
        
        # Add conditional edge from decide
        self.add_conditional_edge(
            workflow_id, 
            "decide", 
            should_continue,
            {
                "think": "think",
                "end": END
            }
        )
        
        # Set entry point
        self.set_entry_point(workflow_id, "think")
        
        return workflow_id
    
    def create_multi_agent_workflow(self, workflow_id: str,
                                   agent_roles: List[str]) -> str:
        """
        Create multi-agent collaborative workflow
        agent_roles: List of agent role names (e.g., ["researcher", "writer", "critic"])
        """
        workflow = self.create_workflow(workflow_id)
        
        # Create node for each agent
        for role in agent_roles:
            def agent_node(state: AgentState, agent_role=role) -> AgentState:
                """Agent performs its role"""
                messages = state["messages"]
                
                prompt = ChatPromptTemplate.from_messages([
                    ("system", f"You are a {agent_role}. Perform your role based on the current context."),
                    ("human", "Task: {task}\nContext: {context}")
                ])
                
                response = self.model.invoke(
                    prompt.format_messages(
                        task=state["current_task"],
                        context=str(state["context"])
                    )
                )
                
                messages.append(AIMessage(content=f"[{agent_role}]: {response.content}"))
                state["results"][agent_role] = response.content
                
                return state
            
            self.add_node(workflow_id, role, agent_node)
        
        # Connect agents sequentially
        for i in range(len(agent_roles) - 1):
            self.add_edge(workflow_id, agent_roles[i], agent_roles[i + 1])
        
        # Last agent goes to END
        self.add_edge(workflow_id, agent_roles[-1], END)
        
        # Set entry point
        self.set_entry_point(workflow_id, agent_roles[0])
        
        return workflow_id
    
    def create_parallel_workflow(self, workflow_id: str,
                                parallel_tasks: List[str]) -> str:
        """
        Create workflow with parallel execution branches
        """
        workflow = self.create_workflow(workflow_id)
        
        def start_node(state: AgentState) -> AgentState:
            """Initialize parallel execution"""
            state["context"]["parallel_results"] = {}
            return state
        
        def merge_node(state: AgentState) -> AgentState:
            """Merge results from parallel branches"""
            results = state["context"].get("parallel_results", {})
            state["results"]["merged"] = results
            return state
        
        # Add start and merge nodes
        self.add_node(workflow_id, "start", start_node)
        self.add_node(workflow_id, "merge", merge_node)
        
        # Create parallel task nodes
        for task_name in parallel_tasks:
            def task_node(state: AgentState, task=task_name) -> AgentState:
                """Execute parallel task"""
                response = self.model.invoke(f"Execute task: {task}")
                state["context"]["parallel_results"][task] = response.content
                return state
            
            self.add_node(workflow_id, task_name, task_node)
            self.add_edge(workflow_id, "start", task_name)
            self.add_edge(workflow_id, task_name, "merge")
        
        self.add_edge(workflow_id, "merge", END)
        self.set_entry_point(workflow_id, "start")
        
        return workflow_id
    
    # ==================== WORKFLOW MANAGEMENT ====================
    
    def get_workflow_status(self, workflow_id: str) -> WorkflowStatus:
        """Get current workflow status"""
        return self.workflows.get(workflow_id, {}).get("status", WorkflowStatus.PENDING)
    
    def list_workflows(self) -> List[str]:
        """List all workflow IDs"""
        return list(self.workflows.keys())
    
    def delete_workflow(self, workflow_id: str):
        """Delete workflow"""
        if workflow_id in self.workflows:
            del self.workflows[workflow_id]
            logger.info(f"Workflow deleted: {workflow_id}")
    
    # ==================== HUMAN-IN-THE-LOOP ====================
    
    def create_human_approval_workflow(self, workflow_id: str) -> str:
        """Create workflow with human approval checkpoints"""
        workflow = self.create_workflow(workflow_id)
        
        def process_node(state: AgentState) -> AgentState:
            """Process task"""
            response = self.model.invoke(state["current_task"])
            state["context"]["result"] = response.content
            state["next_action"] = "approval"
            return state
        
        def approval_node(state: AgentState) -> AgentState:
            """Wait for human approval"""
            result = state["context"].get("result", "")
            state["context"]["awaiting_approval"] = True
            state["context"]["approval_content"] = result
            return state
        
        def execute_node(state: AgentState) -> AgentState:
            """Execute after approval"""
            state["results"]["final"] = state["context"]["approval_content"]
            return state
        
        self.add_node(workflow_id, "process", process_node)
        self.add_node(workflow_id, "approval", approval_node)
        self.add_node(workflow_id, "execute", execute_node)
        
        self.add_edge(workflow_id, "process", "approval")
        self.add_edge(workflow_id, "approval", "execute")
        self.add_edge(workflow_id, "execute", END)
        
        self.set_entry_point(workflow_id, "process")
        
        return workflow_id


# Global orchestrator instance
_orchestrator = None

def get_orchestrator(model_name: str = "gpt-4.1-mini") -> LangGraphOrchestrator:
    """Get or create global orchestrator"""
    global _orchestrator
    
    if _orchestrator is None:
        _orchestrator = LangGraphOrchestrator(model_name=model_name)
    
    return _orchestrator
