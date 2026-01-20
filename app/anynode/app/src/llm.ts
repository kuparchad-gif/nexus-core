// This file will contain the abstraction layer for different LLM backends.

export type LLMBackend = 'lm_studio' | 'vllm' | 'ollama';

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface ChatRequest {
  messages: ChatMessage[];
  url: string;
  apiKey?: string;
}

export type ChatFunction = (request: ChatRequest) => Promise<string>;

const openAIChat: ChatFunction = async ({ messages, url, apiKey }) => {
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
// Import the sanitizeApiKey function from a hypothetical 'utils' module
// This function is assumed to sanitize the API key by removing any newline characters
// import { sanitizeApiKey } from './utils';

const openAIChat: ChatFunction = async ({ messages, url, apiKey }) => {
  const sanitizedApiKey = apiKey ? sanitizeApiKey(apiKey) : '';
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${sanitizedApiKey}`,
    },
    body: JSON.stringify({
      messages,
    },
    body: JSON.stringify({
      messages,
      model: 'local-model', // Model name is often required, but can be a placeholder
      temperature: 0.7,
      stream: false,
    }),
  });

  if (!response.ok) {
    throw new Error(`Failed to fetch from OpenAI compatible API: ${response.statusText}`);
  }

  const data = await response.json();
  return data.choices[0]?.message?.content || '';
};

// Implementations for each backend will go here.

const ollamaChat: ChatFunction = async ({ messages, url }) => {
}

export interface ChatRequest {
  messages: ChatMessage[];
  url: string;
  apiKey?: string;
  model?: string; // Added optional model field
}

export type ChatFunction = (request: ChatRequest) => Promise<string>;

const openAIChat: ChatFunction = async ({ messages, url, apiKey, model = 'local-model' }) => {
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey || ''}`,
    },
    body: JSON.stringify({
      messages,
      model, // Use the provided model or default to 'local-model'
      temperature: 0.7,
      stream: false,
    }),
  });

  if (!response.ok) {
    throw new Error(`Failed to fetch from OpenAI compatible API: ${response.statusText}`);
  }

  const data = await response.json();
  return data.choices[0]?.message?.content || '';
};

// Implementations for each backend will go here.

const ollamaChat: ChatFunction = async ({ messages, url, model = 'llama2' }) => {
  const response = await fetch(`${url}/api/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model, // Use the provided model or default to 'llama2'
      messages,
      stream: false,
    }),
  });

  if (!response.ok) {
    throw new Error(`Failed to fetch from Ollama API: ${response.statusText}`);
  }
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: 'llama3', // Or another model name
      messages,
      stream: false,
    }),
  });

  if (!response.ok) {
    throw new Error(`Failed to fetch from Ollama API: ${response.statusText}`);
  }

  const data = await response.json();
export type ChatFunction = (request: ChatRequest) => Promise<string>;

const openAIChat: ChatFunction = async ({ messages, url, apiKey }) => {
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey || ''}`,
      },
      body: JSON.stringify({
        messages,
        model: 'local-model', // Model name is often required, but can be a placeholder
        temperature: 0.7,
        stream: false,
      }),
    });

    if (!response.ok) {
      throw new Error(`Failed to fetch from OpenAI compatible API: ${response.statusText}`);
    }

    const data = await response.json();
    return data.choices[0]?.message?.content || '';
  } catch (error) {
    if (error instanceof TypeError) {
      throw new Error(`Network error: ${error.message}`);
    } else if (error instanceof SyntaxError) {
      throw new Error(`Invalid JSON response: ${error.message}`);
    } else {
      throw error;
    }
  }
};

// Implementations for each backend will go here.

const ollamaChat: ChatFunction = async ({ messages, url }) => {
  try {
    const response = await fetch(`${url}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'llama3', // Or another model name
        messages,
        stream: false,
      }),
    });

    if (!response.ok) {
      throw new Error(`Failed to fetch from Ollama API: ${response.statusText}`);
    }

    const data = await response.json();
    return data.message?.content || '';
  } catch (error) {
    if (error instanceof TypeError) {
      throw new Error(`Network error: ${error.message}`);
    } else if (error instanceof SyntaxError) {
      throw new Error(`Invalid JSON response: ${error.message}`);
    } else {
      throw error;
    }
  }
};

// Factory function to get the correct chat function.
};

// Factory function to get the correct chat function.
export const getChatFunction = (backend: LLMBackend): ChatFunction => {
  switch (backend) {
    case 'lm_studio':
      return openAIChat;
    case 'vllm':
      return openAIChat;
    case 'ollama':
      return ollamaChat;
    default:
      throw new Error(`Unsupported backend: ${backend}`);
  }
};
