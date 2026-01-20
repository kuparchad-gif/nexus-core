


import * as geminiProvider from './providers/geminiProvider';
// Import other providers here as they are implemented
// import * as ollamaProvider from './providers/ollamaProvider';

const PROVIDERS = {
    gemini: geminiProvider,
    // ollama: ollamaProvider, // Uncomment when ollamaProvider is implemented
};

const CURRENT_PROVIDER = PROVIDERS.gemini;
const FALLBACK_PROVIDER = null; 

async function withRetryAndSwap<T>(
    apiCall: (provider: any) => Promise<T>,
): Promise<T> {
    let lastError: Error | unknown = new Error("API call failed after all retries and fallbacks.");

    try {
        return await apiCall(CURRENT_PROVIDER);
    } catch (error) {
        console.warn(`API call with primary provider failed. Error: ${error instanceof Error ? error.message : String(error)}`);
        lastError = error;
    }

    if (FALLBACK_PROVIDER) {
        console.log("Switching to fallback provider...");
        try {
            return await apiCall(FALLBACK_PROVIDER);
        } catch (error) {
            console.error(`API call with fallback provider also failed. Error: ${error instanceof Error ? error.message : String(error)}`);
            lastError = error;
        }
    }
    
    throw lastError;
}

export async function analyzeMedia(file: Express.Multer.File): Promise<string> {
    return withRetryAndSwap((provider) => provider.analyzeMedia(file));
}

export async function generateAssets(prompt: string, sessionId: string, file?: Express.Multer.File) {
    return withRetryAndSwap((provider) => provider.generateAssets(prompt, sessionId, file));
}

export async function generateProp(prompt: string, sessionId: string) {
    return withRetryAndSwap((provider) => provider.generateProp(prompt, sessionId));
}

export async function editImage(imageUrl: string, prompt: string): Promise<string> {
    return withRetryAndSwap((provider) => provider.editImage(imageUrl, prompt));
}

export async function applyClothing(imageUrl: string, prompt: string, clothingFile?: Express.Multer.File): Promise<string> {
    return withRetryAndSwap((provider) => provider.applyClothing(imageUrl, prompt, clothingFile));
}

export async function getChatReply(sessionId: string, mode: 'assistant' | 'avatar' | 'support', history: any[]): Promise<string> {
    return withRetryAndSwap((provider) => provider.getChatReply(sessionId, mode, history));
}

export async function generateSocialPost(description: string): Promise<string> {
    return withRetryAndSwap((provider) => provider.generateSocialPost(description));
}

export async function generateLore(description: string, topic: string, sessionId: string): Promise<string> {
    return withRetryAndSwap((provider) => provider.generateLore(description, topic, sessionId));
}