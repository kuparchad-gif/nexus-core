import { GoogleGenAI, Modality } from "@google/genai";
import { toBase64 } from "../utils/fileUtils";
import { getApiKey } from "../utils/apiKey";

// A single, initialized AI client instance.
let ai: GoogleGenAI | null = null;

function getAiClient(): GoogleGenAI {
    if (!ai) {
        const apiKey = getApiKey();
        if (!apiKey) {
            throw new Error("API_KEY is not available. Please check your environment configuration.");
        }
        ai = new GoogleGenAI({ apiKey });
    }
    return ai;
}

// Define model configurations for easier management and fallback.
// This allows for "hot-swapping" a model if the primary one fails.
const MODEL_CONFIG = {
    analysis: { primary: 'gemini-2.5-flash', fallback: 'gemini-2.5-flash' }, 
    description: { primary: 'gemini-2.5-flash', fallback: 'gemini-2.5-flash' },
    social: { primary: 'gemini-2.5-flash', fallback: 'gemini-2.5-flash' },
    imageGen: { primary: 'imagen-4.0-generate-001', fallback: 'imagen-4.0-generate-001' }, 
    videoGen: { primary: 'veo-2.0-generate-001', fallback: 'veo-2.0-generate-001' },
    imageEdit: { primary: 'gemini-2.5-flash-image-preview', fallback: 'gemini-2.5-flash-image-preview' },
};

/**
 * Higher-Order Function for AI Model Resilience: `withRetry`
 *
 * This function wraps an AI API call to provide two key features for robustness:
 * 1.  Automatic Retries: It retries the request upon failure using an exponential backoff strategy
 *     to avoid overwhelming the service.
 * 2.  Model "Hot-Swapping": If the primary model fails (e.g., due to downtime or deprecation),
 *     it automatically "hot-swaps" to a pre-configured fallback model and retries the request.
 *
 * This makes the application more resilient and adaptable to transient API issues.
 */
async function withRetry<T>(
    apiCall: (model: string) => Promise<T>, 
    modelConfig: { primary: string, fallback: string }, 
    retries = 2
): Promise<T> {
    let lastError: Error | unknown = new Error("API call failed after all retries.");
    
    // First attempt with the primary model
    try {
        return await apiCall(modelConfig.primary);
    } catch (error) {
        console.warn(`API call with primary model ${modelConfig.primary} failed. Error: ${error instanceof Error ? error.message : String(error)}`);
        lastError = error;
    }

    // Retry logic with the fallback model
    for (let i = 0; i < retries; i++) {
        try {
            console.log(`Retrying with fallback model ${modelConfig.fallback}, attempt ${i + 1}...`);
            const delay = Math.pow(2, i) * 1000;
            await new Promise(res => setTimeout(res, delay));
            return await apiCall(modelConfig.fallback);
        } catch (error) {
            console.warn(`Fallback attempt ${i + 1} failed. Error: ${error instanceof Error ? error.message : String(error)}`);
            lastError = error;
        }
    }

    throw lastError;
}

export async function analyzeMedia(file: File): Promise<string> {
    const client = getAiClient();
    const base64Data = await toBase64(file) as string;
    const mimeType = file.type;

    const VTT_PROMPT = "Analyze the provided image or video of a character. Provide a detailed, one-paragraph description suitable for a text-to-video generation prompt. Focus on appearance, attire, style, and mood. For example: 'A cinematic shot of a futuristic knight with glowing blue cybernetic armor, standing in a neon-lit alley. The mood is heroic and slightly gritty.'";

    try {
        const apiCall = (model: string) => client.models.generateContent({
            model,
            contents: { parts: [{ inlineData: { mimeType, data: base64Data } }, { text: VTT_PROMPT }] },
        });
        const response = await withRetry(apiCall, MODEL_CONFIG.analysis);
        return response.text;
    } catch (error) {
         console.error("Error in Gemini analyzeMedia service:", error);
         throw new Error(`Failed to analyze media: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
}

export async function editImage(imageUrl: string, userPrompt: string): Promise<string> {
    const client = getAiClient();
    const base64Response = await fetch(imageUrl);
    const blob = await base64Response.blob();
    const base64Data = await toBase64(blob) as string;
    const mimeType = blob.type;
    
    try {
        const apiCall = (model: string) => client.models.generateContent({
            model,
            contents: { parts: [{ inlineData: { data: base64Data, mimeType } }, { text: userPrompt }] },
            config: { responseModalities: [Modality.IMAGE, Modality.TEXT] },
        });
        const response = await withRetry(apiCall, MODEL_CONFIG.imageEdit);

        for (const part of response.candidates[0].content.parts) {
            if (part.inlineData) {
              const newBase64ImageBytes: string = part.inlineData.data;
              return `data:${part.inlineData.mimeType};base64,${newBase64ImageBytes}`;
            }
        }
        throw new Error("No image was returned from the edit request.");
    } catch (error) {
        console.error("Error in Gemini editImage service:", error);
        throw new Error(`Failed to edit image: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
}

/**
 * Toolset Orchestration: `generateAsset`
 *
 * This function acts as an intelligent agent or an MCP (Master Control Program) tool.
 * It orchestrates a sequence of calls to different, specialized AI models to achieve a complex goal.
 * This "tool-chaining" pattern allows the application to perform sophisticated tasks that a single
 * model call could not.
 *
 * Workflow:
 * 1.  (Tool 1 - Analysis/Description) Use a language model to refine the user's prompt into a
 *     detailed description.
 * 2.  (Tool 2 - Image Generation) Use an image generation model to create a character sheet
 *     based on the detailed description.
 * 3.  (Tool 3 - Video Generation) Use a video generation model to create a cinematic turntable
 *     animation, also based on the detailed description.
 */
export async function generateAsset(userPrompt: string, file: File | null): Promise<{ videoUrl: string, description: string, characterSheetUrl: string }> {
    const client = getAiClient();
    try {
        // Step 1: Generate a detailed description
        let finalDescriptionPrompt = `Based on the user's request, create a detailed, descriptive prompt for a 3D modeling AI to generate a photorealistic, full-body 3D avatar. The user's request is: "${userPrompt}"`;
        let contents: any = { parts: [{ text: finalDescriptionPrompt }] };

        if (file) {
             const base64Data = await toBase64(file) as string;
             const mimeType = file.type;
             contents = { parts: [{ inlineData: { mimeType, data: base64Data } }, { text: `Analyze the provided media and use it as a primary reference. ${finalDescriptionPrompt}` }] }
        }
        const descriptionApiCall = (model: string) => client.models.generateContent({ model, contents });
        const descriptionResponse = await withRetry(descriptionApiCall, MODEL_CONFIG.description);
        const generatedDescription = descriptionResponse.text;

        // Step 2: Generate the character sheet image
        const characterSheetPrompt = `Generate a character sheet for a 3D model. The image must feature multiple angles of the character (front, side, and back view) in a T-pose against a neutral, plain gray background. The style should be photorealistic. Use this detailed description: "${generatedDescription}"`;
        const imageApiCall = (model: string) => client.models.generateImages({
            model,
            prompt: characterSheetPrompt,
            config: { numberOfImages: 1, outputMimeType: 'image/png', aspectRatio: '16:9' },
        });
        const imageResponse = await withRetry(imageApiCall, MODEL_CONFIG.imageGen);
        const base64ImageBytes = imageResponse.generatedImages[0].image.imageBytes;
        const characterSheetUrl = `data:image/png;base64,${base64ImageBytes}`;
        
        // Step 3: Generate the video
        const videoGenerationPrompt = `Generate a high-quality, cinematic 3D video of a character. The character should be in a neutral environment, slowly rotating to show all angles (a turntable animation). Based on this description: ${generatedDescription}`;
        const videoApiCall = (model: string) => client.models.generateVideos({ model, prompt: videoGenerationPrompt, config: { numberOfVideos: 1 } });
        let operation = await withRetry(videoApiCall, MODEL_CONFIG.videoGen, 1);

        while (!operation.done) {
            await new Promise(resolve => setTimeout(resolve, 10000));
            operation = await client.operations.getVideosOperation({operation: operation});
        }
        const downloadLink = operation.response?.generatedVideos?.[0]?.video?.uri;

        if (downloadLink && characterSheetUrl) {
            return { videoUrl: downloadLink, description: generatedDescription, characterSheetUrl };
        } else {
            throw new Error('Video or character sheet generation failed.');
        }

    } catch (error) {
        console.error("Error in Gemini generateAsset service:", error);
        throw new Error(`Failed to generate asset: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
}

export async function generateSocialPost(characterDescription: string, characterSheetUrl: string): Promise<string> {
    const client = getAiClient();
    const imageBlob = await (await fetch(characterSheetUrl)).blob();
    const base64Data = await toBase64(imageBlob) as string;

    const SOCIAL_PROMPT = `You are a social media manager for a digital artist. 
    Analyze the provided image and the character description.
    Write an exciting and engaging social media post to showcase this new creation.
    - Keep it concise (2-3 sentences).
    - Include a call to action, like asking a question.
    - Add 3-5 relevant and popular hashtags.
    
    Character Description: "${characterDescription}"`;

    try {
        const apiCall = (model: string) => client.models.generateContent({
            model,
            contents: { parts: [{ inlineData: { mimeType: 'image/png', data: base64Data } }, { text: SOCIAL_PROMPT }] },
        });
        const response = await withRetry(apiCall, MODEL_CONFIG.social);
        return response.text;
    } catch (error) {
         console.error("Error in Gemini generateSocialPost service:", error);
         throw new Error(`Failed to generate social post: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
}