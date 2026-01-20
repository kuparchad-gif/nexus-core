// Fix: Use a consistent express import to ensure correct type resolution.
import express from 'express';
import * as aiService from '../services/aiService';

const asyncHandler = (fn: (req: express.Request, res: express.Response, next: express.NextFunction) => Promise<void>) => 
    (req: express.Request, res: express.Response, next: express.NextFunction) => {
        Promise.resolve(fn(req, res, next)).catch(next);
    };

export const getHealth = (req: express.Request, res: express.Response) => {
    res.json({ status: 'ok' });
};

export const analyzeMedia = asyncHandler(async (req: express.Request, res: express.Response) => {
    if (!req.file) {
        res.status(400).json({ message: 'No file uploaded.' });
        return;
    }
    const description = await aiService.analyzeMedia(req.file);
    res.json({ description });
});

export const generateAssets = asyncHandler(async (req: express.Request, res: express.Response) => {
    const { prompt, sessionId } = req.body;
    if (!prompt || !sessionId) {
        res.status(400).json({ message: 'Prompt and sessionId are required.' });
        return;
    }
    const assets = await aiService.generateAssets(prompt, sessionId, req.file);
    res.json(assets);
});

export const generateProp = asyncHandler(async (req: express.Request, res: express.Response) => {
    const { prompt, sessionId } = req.body;
    if (!prompt || !sessionId) {
        res.status(400).json({ message: 'Prompt and sessionId are required.' });
        return;
    }
    const prop = await aiService.generateProp(prompt, sessionId);
    res.json(prop);
});

export const editImage = asyncHandler(async (req: express.Request, res: express.Response) => {
    const { imageUrl, prompt } = req.body;
    if (!imageUrl || !prompt) {
        res.status(400).json({ message: 'imageUrl and prompt are required.' });
        return;
    }
    const newImageUrl = await aiService.editImage(imageUrl, prompt);
    res.json({ newImageUrl });
});

export const applyClothing = asyncHandler(async (req: express.Request, res: express.Response) => {
    const { imageUrl, prompt } = req.body;
    if (!imageUrl || !prompt) {
        res.status(400).json({ message: 'imageUrl and prompt are required.' });
        return;
    }
    const newImageUrl = await aiService.applyClothing(imageUrl, prompt, req.file);
    res.json({ newImageUrl });
});

export const handleChat = asyncHandler(async (req: express.Request, res: express.Response) => {
    const { sessionId, mode, history } = req.body;
    if (!sessionId || !mode || !history) {
        res.status(400).json({ message: "sessionId, mode, and history are required." });
        return;
    }
    const reply = await aiService.getChatReply(sessionId, mode, history);
    res.json({ reply });
});

export const generateSocialPost = asyncHandler(async (req: express.Request, res: express.Response) => {
    const { description } = req.body;
    if (!description) {
        res.status(400).json({ message: "Character description is required." });
        return;
    }
    const post = await aiService.generateSocialPost(description);
    res.json({ post });
});

export const generateLore = asyncHandler(async (req: express.Request, res: express.Response) => {
    const { description, topic, sessionId } = req.body;
    if (!description || !topic || !sessionId) {
        res.status(400).json({ message: "Description, topic, and sessionId are required." });
        return;
    }
    const lore = await aiService.generateLore(description, topic, sessionId);
    res.json({ lore });
});

export const getAvailablePlugins = asyncHandler(async (req: express.Request, res: express.Response) => {
    // In a real application, this list would come from a database or configuration file.
    res.json([
        { id: 'docker-dash', name: 'Docker Dashboard', description: 'Monitor and manage Docker containers.', url: 'http://localhost:9000' },
        { id: 'plex-dash', name: 'Plex Media Server', description: 'Manage your Plex media library.', url: 'http://localhost:32400/web' },
        { id: 'code-server', name: 'Code Editor (IDE)', description: 'An onboard VS Code IDE for platform development.', url: 'http://localhost:8080' }
    ]);
});

export const startTrainingJob = asyncHandler(async (req: express.Request, res: express.Response) => {
    const { assets, parameters } = req.body;
    console.log("[Training Job Received]:", { assets, parameters });
    // This is the "wire". In a real implementation, this would trigger a long-running job.
    res.status(202).json({ message: "Training job accepted." });
});