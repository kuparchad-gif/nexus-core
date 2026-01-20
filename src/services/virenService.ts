

import { v4 as uuidv4 } from 'uuid';
// FIX: Removed 'Product' from import as it is not defined in '../types'.
import { Agent, AgentStatus, TaskStatus, Tool, AgentType } from '../types';

// A helper to simulate network delay and potential errors
const apiFetch = async <T>(action: () => T, delay: number = 500): Promise<T> => {
    await new Promise(res => setTimeout(res, delay));
    // Simulate a random connection error
    if (Math.random() < 0.05) {
        throw new Error("Failed to connect to the VIREN service. Please check the connection.");
    }
    return action();
};

// --- MOCK DATABASE ---

const MOCK_TOOLS: Tool[] = [
    { id: 'web_browser', name: 'Web Browser', description: 'Access websites to gather information.', tags: ['web'], input_schema: { "url": "string" } },
    { id: 'shell', name: 'Shell Terminal', description: 'Execute shell commands in a sandboxed environment.', tags: ['system', 'code'], input_schema: { "command": "string" } },
    { id: 'memory', name: 'Long-term Memory', description: 'Store and retrieve information from memory.', tags: ['memory'], input_schema: { "operation": "read|write", "key": "string", "value": "any" } },
    { id: 'human_approval', name: 'Human Approval', description: 'Request human approval for a critical step.', tags: ['system'], input_schema: { "prompt": "string" } },
    { id: 'ecommerce_manager', name: 'E-commerce Manager', description: 'Manage products and orders on platforms like Shopify.', tags: ['finance'], input_schema: { "action": "list_products|update_inventory" } },
    { id: 'email_account_creator', name: 'Email Account Creator', description: 'Creates and configures new email accounts.', tags: ['system'], input_schema: { "provider": "gmail|outlook", "username": "string" } },
];

let MOCK_AGENTS: Agent[] = [
    { 
        id: uuidv4(), 
        goal: 'Research the market for AI-powered developer tools and write a summary.', 
        status: AgentStatus.RUNNING, 
        tasks: [
            {id: 't1-1', description: 'Search Google for "AI developer tools market analysis"', status: TaskStatus.COMPLETED, type: 'BROWSER', output: 'Found 5 relevant articles.'}, 
            {id: 't1-2', description: 'Read and summarize the top 3 articles.', status: TaskStatus.IN_PROGRESS, type: 'BROWSER'},
            {id: 't1-3', description: 'Synthesize summaries into a final report.', status: TaskStatus.PENDING, type: 'MEMORY'},
        ], 
        logs: ['Agent initialized.', 'Starting market research.', 'Accessing web browser tool...'], 
        tools: ['Web Browser', 'Long-term Memory'], 
        chatHistory: [], 
        createdAt: new Date(Date.now() - 3600000).toISOString(),
        type: 'STANDARD',
        connectionStatus: 'CONNECTED',
        cpuAllocation: 50, ramAllocation: 30, storageAllocation: 10, gpuAllocation: 0,
        currentTaskIndex: 1,
        financials: { revenue: 0, profit: 0, goal: 0 }
    },
    { 
        id: uuidv4(), 
        goal: 'Create a new Shopify store and list 3 products.', 
        status: AgentStatus.AWAITING_USER_INPUT, 
        tasks: [
             {id: 't2-1', description: 'Create a new Shopify account under the name "Aethereal Wares".', status: TaskStatus.COMPLETED, type: 'BROWSER', output: 'Account created successfully.'}, 
             {id: 't2-2', description: 'List "Nexus T-Shirt" product.', status: TaskStatus.COMPLETED, type: 'BROWSER', output: 'Product listed.'},
             {id: 't2-3', description: 'Authorize payment gateway integration.', status: TaskStatus.PENDING, type: 'APPROVAL'},
        ], 
        logs: ['Financial agent activated.', 'Accessing E-commerce tools...'], 
        tools: ['E-commerce Manager', 'Web Browser', 'Human Approval'], 
        chatHistory: [], 
        createdAt: new Date(Date.now() - 7200000).toISOString(),
        type: 'FINANCE',
        connectionStatus: 'NEEDS_CONFIG',
        cpuAllocation: 75, ramAllocation: 55, storageAllocation: 20, gpuAllocation: 10,
        currentTaskIndex: 2,
        financials: { revenue: 0, profit: 0, goal: 1500 }
    },
];

// --- API FUNCTIONS ---

export const getAgents = (): Promise<Agent[]> => apiFetch(() => JSON.parse(JSON.stringify(MOCK_AGENTS)));

export const createAgent = (goal: string, tools: string[], agentType: AgentType, financialGoal: number): Promise<Agent> => apiFetch(() => {
    const newAgent: Agent = {
        id: uuidv4(),
        goal,
        tools,
        type: agentType,
        status: AgentStatus.PLANNING,
        tasks: [{ id: uuidv4(), description: 'Formulate a plan to achieve the goal.', status: TaskStatus.IN_PROGRESS, type: 'MEMORY' }],
        logs: ['Agent created.', 'Initiating planning phase.'],
        chatHistory: [],
        createdAt: new Date().toISOString(),
        connectionStatus: 'CONNECTED',
        cpuAllocation: 30, ramAllocation: 20, storageAllocation: 5, gpuAllocation: 0,
        currentTaskIndex: 0,
        financials: { revenue: 0, profit: 0, goal: agentType === 'FINANCE' ? financialGoal : 0 }
    };
    MOCK_AGENTS.unshift(newAgent);
    return newAgent;
}, 1000);

export const getTools = (): Promise<Tool[]> => apiFetch(() => JSON.parse(JSON.stringify(MOCK_TOOLS)), 200);