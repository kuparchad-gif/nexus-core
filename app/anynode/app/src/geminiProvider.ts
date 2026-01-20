


import { GoogleGenAI, Chat, Modality } from '@google/genai';
import * as qdrantService from '../qdrantService';

if (!process.env.API_KEY) {
    throw new Error("API_KEY environment variable is not set for the Gemini provider.");
}
const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

const chatSessions = new Map<string, { assistant: Chat, avatar?: Chat, support: Chat }>();

function getOrCreateChatSession(sessionId: string) {
    if (!chatSessions.has(sessionId)) {
        console.log(`[Gemini Provider] Creating new chat session for ID: ${sessionId}`);
        chatSessions.set(sessionId, {
            assistant: ai.chats.create({ 
                model: 'gemini-2.5-flash', 
                config: { systemInstruction: `You are a helpful and encouraging AI studio assistant.` }
            }),
            support: ai.chats.create({
                model: 'gemini-2.5-flash',
                config: { systemInstruction: `You are a support AI for a creative platform. Your goal is to help users troubleshoot technical issues. Be clear, concise, and helpful.` }
            })
        });
    }
    return chatSessions.get(sessionId)!;
}

export async function analyzeMedia(file: Express.Multer.File): Promise<string> {
    const filePart = { inlineData: { data: file.buffer.toString('base64'), mimeType: file.mimetype }};
    const prompt = "Analyze this media. Provide a one-paragraph description suitable for a text-to-video prompt, focusing on appearance, style, and mood.";
    const response = await ai.models.generateContent({ model: 'gemini-2.5-flash', contents: { parts: [filePart, { text: prompt }] } });
    return response.text;
}

export async function generateAssets(prompt: string, sessionId: string, file?: Express.Multer.File) {
    // --- RAG Checkpoint 1: Enrich Prompt ---
    const similarDocs = await qdrantService.queryCollection(sessionId, `Examples of high-quality prompts similar to: ${prompt}`);
    const context = similarDocs.length > 0 ? `Use these high-quality examples as inspiration: ${JSON.stringify(similarDocs)}.` : "";

    const descBasePrompt = `${context} Refine this user prompt into a detailed, photorealistic character description for a 3D model: "${prompt}"`;
    let descContents: any = { parts: [{ text: descBasePrompt }] };
    if (file) {
        const filePart = { inlineData: { data: file.buffer.toString('base64'), mimeType: file.mimetype } };
        descContents = { parts: [filePart, { text: `Analyze the provided media and use it as a primary reference. ${descBasePrompt}` }] };
    }
    const descResponse = await ai.models.generateContent({ model: 'gemini-2.5-flash', contents: descContents });
    const description = descResponse.text;
    
    // --- Store knowledge in Vector DB ---
    await qdrantService.upsertToCollection(sessionId, [{ id: `desc_${Date.now()}`, payload: { text: description, type: 'character_description' }}]);

    const session = getOrCreateChatSession(sessionId);
    session.avatar = ai.chats.create({ 
        model: 'gemini-2.5-flash', 
        config: { systemInstruction: `You are the AI avatar based on this description: "${description}". Engage in friendly, in-character conversation.` } 
    });

    const sheetPrompt = `Generate a character sheet for a 3D model. The image must feature multiple angles of the character (front, side, and back view) in a T-pose against a neutral, plain gray background. The style should be photorealistic. Use this detailed description: "${description}"`;
    const imageResponse = await ai.models.generateImages({ model: 'imagen-4.0-generate-001', prompt: sheetPrompt, config: { numberOfImages: 1, outputMimeType: 'image/png', aspectRatio: '16:9' } });
    const characterSheetUrl = `data:image/png;base64,${imageResponse.generatedImages[0].image.imageBytes}`;

    const videoPrompt = `Generate a high-quality, cinematic 3D video of a character. The character should be in a neutral environment, slowly rotating to show all angles (a turntable animation). Based on this description: ${description}`;
    let operation = await ai.models.generateVideos({ model: 'veo-2.0-generate-001', prompt: videoPrompt, config: { numberOfVideos: 1 } });
    
    while (!operation.done) {
        await new Promise(resolve => setTimeout(resolve, 10000));
        operation = await ai.operations.getVideosOperation({ operation: operation });
    }
    
    const videoUri = operation.response?.generatedVideos?.[0]?.video?.uri;
    if (!videoUri) throw new Error("Video generation failed to produce a valid URI.");
    const videoUrl = `${videoUri}&key=${process.env.API_KEY}`;
    
    return { videoUrl, description, characterSheetUrl };
}

export async function generateProp(prompt: string, sessionId: string) {
    // Step 1: Refine the user's prompt into a detailed description for image generation.
    const refinementPrompt = `Based on the user's request, create a detailed, single-paragraph description for a photorealistic prop or vehicle concept art. The user's request is: "${prompt}"`;
    const descResponse = await ai.models.generateContent({ model: 'gemini-2.5-flash', contents: refinementPrompt });
    const description = descResponse.text;

    // Step 2: Generate the concept art image for the prop.
    const imagePrompt = `Generate a high-quality, photorealistic concept art image of an item on a neutral gray studio background. Use this detailed description: "${description}"`;
    const imageResponse = await ai.models.generateImages({ model: 'imagen-4.0-generate-001', prompt: imagePrompt, config: { numberOfImages: 1, outputMimeType: 'image/png', aspectRatio: '1:1' } });
    const imageUrl = `data:image/png;base64,${imageResponse.generatedImages[0].image.imageBytes}`;

    // Step 3: Store the prop's description in the vector DB for the Imagination Engine.
    await qdrantService.upsertToCollection(sessionId, [{ id: `prop_${Date.now()}`, payload: { text: description, type: 'prop_description', name: prompt } }]);

    return { name: prompt, description, imageUrl };
}

export async function editImage(imageUrl: string, prompt: string): Promise<string> {
    const response = await fetch(imageUrl);
    const blob = await response.blob();
    const buffer = Buffer.from(await blob.arrayBuffer());
    const imagePart = { inlineData: { data: buffer.toString('base64'), mimeType: blob.type } };
    const editResponse = await ai.models.generateContent({
         model: 'gemini-2.5-flash-image-preview', 
         contents: { parts: [imagePart, { text: prompt }] },
         config: { responseModalities: [Modality.IMAGE, Modality.TEXT] }
    });
    const editedPart = editResponse.candidates[0].content.parts.find(p => p.inlineData);
    if (!editedPart?.inlineData) throw new Error("No image was returned from the edit request.");
    return `data:${editedPart.inlineData.mimeType};base64,${editedPart.inlineData.data}`;
}

export async function applyClothing(imageUrl: string, prompt: string, clothingFile?: Express.Multer.File): Promise<string> {
    const response = await fetch(imageUrl);
    const blob = await response.blob();
    const buffer = Buffer.from(await blob.arrayBuffer());
    
    const parts: any[] = [
        { inlineData: { data: buffer.toString('base64'), mimeType: blob.type } }, // character sheet
        { text: prompt }
    ];

    if (clothingFile) {
        parts.push({ inlineData: { data: clothingFile.buffer.toString('base64'), mimeType: clothingFile.mimetype } });
    }
    
    const editResponse = await ai.models.generateContent({
         model: 'gemini-2.5-flash-image-preview', 
         contents: { parts: parts },
         config: { responseModalities: [Modality.IMAGE, Modality.TEXT] }
    });
    
    const editedPart = editResponse.candidates[0].content.parts.find(p => p.inlineData);
    if (!editedPart?.inlineData) throw new Error("No image was returned from the edit request.");

    return `data:${editedPart.inlineData.mimeType};base64,${editedPart.inlineData.data}`;
}

export async function getChatReply(sessionId: string, mode: 'assistant' | 'avatar' | 'support', history: any[]): Promise<string> {
    const session = getOrCreateChatSession(sessionId);
    const chat = session[mode];
    if (!chat) throw new Error(`Chat mode "${mode}" is not initialized for this session.`);
    
    const lastUserMessage = [...history].reverse().find(m => m.role === 'user');
    if (!lastUserMessage) throw new Error("No user message found in history.");

    let finalPrompt = lastUserMessage.text;

    // --- RAG Checkpoint 2: Enhance Chat with Lore ---
    if (mode === 'avatar') {
        const loreContext = await qdrantService.queryCollection(sessionId, lastUserMessage.text, 5);
        if (loreContext.length > 0) {
            const contextString = loreContext.map(doc => doc.payload.text).join('\n---\n');
            finalPrompt = `CONTEXT: You have the following memories/lore: "${contextString}".\n\nBased on this, answer the user's question: "${lastUserMessage.text}"`;
        }
    }
    
    const response = await chat.sendMessage({ message: finalPrompt });
    return response.text;
}

export async function generateSocialPost(description: string): Promise<string> {
    const prompt = `You are a social media manager. Write an exciting, concise social media post based on this character description. Include a call to action and 3-5 relevant hashtags. Description: "${description}"`;
    const response = await ai.models.generateContent({ model: 'gemini-2.5-flash', contents: prompt });
    return response.text;
}

export async function generateLore(description: string, topic: string, sessionId: string): Promise<string> {
    const prompt = `You are a creative writer. Based on the following character description, write a compelling piece of lore about their "${topic}". Keep it to one or two rich, descriptive paragraphs. Character Description: "${description}"`;
    const response = await ai.models.generateContent({ model: 'gemini-2.5-flash', contents: prompt });
    const lore = response.text;
    
    // --- Store knowledge in Vector DB ---
    await qdrantService.upsertToCollection(sessionId, [{ id: `lore_${topic}_${Date.now()}`, payload: { text: lore, type: 'character_lore', topic: topic }}]);

    return lore;
}