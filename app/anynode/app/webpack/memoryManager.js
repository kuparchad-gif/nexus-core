import { storeMessage, findSimilarMessages } from './qdrantClient.js';
import { logMessage, logSystemEvent } from './lokiLogger.js';

// Mock embedding function - replace with OpenAI or similar
async function createEmbedding(text) {
  // Placeholder: return random 1536-dimensional vector
  return Array.from({ length: 1536 }, () => Math.random() - 0.5);
}

export async function processMessage(text, user) {
  try {
    // 1. Generate embedding
    const embedding = await createEmbedding(text);
    
    // 2. Store in Qdrant
    const embeddingId = await storeMessage(text, user, embedding);
    
    // 3. Log to Loki
    await logMessage(text, user, embeddingId);
    
    // 4. Find similar memories
    const similarMessages = await findSimilarMessages(embedding, 3);
    
    // 5. Log retrieval
    await logSystemEvent("memory_retrieval", {
      user,
      similarCount: similarMessages.length,
      embeddingId
    });
    
    return {
      embeddingId,
      similarMessages,
      context: similarMessages.map(m => m.text).join('\n')
    };
    
  } catch (error) {
    await logSystemEvent("memory_error", { user, error: error.message });
    throw error;
  }
}

export async function getConversationContext(text, user) {
  const embedding = await createEmbedding(text);
  const similar = await findSimilarMessages(embedding, 5);
  
  return {
    recentMemories: similar,
    contextPrompt: `Previous relevant conversations:\n${similar.map(m => 
      `[${m.user}]: ${m.text}`
    ).join('\n')}\n\nCurrent message: ${text}`
  };
}