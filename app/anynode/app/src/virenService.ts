import { v4 as uuidv4 } from 'uuid';
import { Agent, AgentStatus, TaskStatus, Tool, AgentType } from '../types';

const apiFetch = async <T>(url: string, method: string = 'GET', body?: any): Promise<T> => {
    const response = await fetch(`http://localhost:8080/${url}`, {
        method,
        headers: { 'Content-Type': 'application/json' },
        body: body ? JSON.stringify(body) : undefined
    });
    if (!response.ok) throw new Error(`Failed to connect to Viren: ${response.statusText}`);
    return response.json();
};

export const getAgents = async (): Promise<Agent[]> => {
    return apiFetch('agents');
};

export const createAgent = async (goal: string, tools: string[], agentType: AgentType, financialGoal: number, modelEndpointId: string): Promise<Agent> => {
    return apiFetch('agents', 'POST', { goal, tools, agentType, financialGoal, modelEndpointId });
};

export const getTools = async (): Promise<Tool[]> => {
    return apiFetch('tools');
};

export const runPsCommand = async (command: string): Promise<any> => {
    return apiFetch('run_ps', 'POST', { command });
};

export const browseWeb = async (url: string): Promise<any> => {
    return apiFetch('browse_web', 'POST', { url });
};

export const systemCmd = async (command: string): Promise<any> => {
    return apiFetch('system_cmd', 'POST', { command });
};

export const sendMessage = async (message: string): Promise<string> => {
    const response = await fetch('http://localhost:8080/infer_sse', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: message })
    });
    const reader = response.body?.getReader();
    if (!reader) throw new Error('No response stream');
    let result = '';
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        result += new TextDecoder().decode(value);
    }
    return result;
};