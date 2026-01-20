import { Router } from 'express';
import multer from 'multer';
import * as assetController from '../controllers/assetController';

const router = Router();
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 50 * 1024 * 1024 } }); // 50MB limit

// --- HEALTH CHECK ---
router.get('/health', assetController.getHealth);

// --- ASSET & CHAT ROUTES ---
router.post('/analyze', upload.single('file'), assetController.analyzeMedia);
router.post('/generate', upload.single('file'), assetController.generateAssets);
router.post('/generate-prop', assetController.generateProp);
router.post('/edit-image', assetController.editImage);
// Fix: Add the new route for applying clothing
router.post('/apply-clothing', upload.single('file'), assetController.applyClothing);
router.post('/chat', assetController.handleChat);
router.post('/generate-social-post', assetController.generateSocialPost);
router.post('/generate-lore', assetController.generateLore);

export default router;