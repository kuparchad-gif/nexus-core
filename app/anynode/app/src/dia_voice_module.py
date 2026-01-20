import os
import subprocess
import torch
from transformers import AutoProcessor, AutoModelForTextToSpeech
import soundfile as sf
import numpy as np
from datetime import datetime
import asyncio
import websockets
import json

class DiaVoiceModule:
    """Dia-1.6B Voice Module for LILLITH's Text-to-Speech"""
    
    def __init__(self, model_path="./models/Dia-1.6B"):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.voice_cache = {}
        
    def download_model(self):
        """Download Dia-1.6B from HuggingFace"""
        if not os.path.exists(self.model_path):
            print("🔄 Downloading Dia-1.6B Voice Model...")
            os.makedirs("./models", exist_ok=True)
            
            subprocess.run([
                "git", "clone", 
                "https://huggingface.co/nari-labs/Dia-1.6B",
                self.model_path
            ], check=True)
            
            print("✅ Dia-1.6B Voice Model downloaded!")
        else:
            print("✅ Dia-1.6B Voice Model ready")
    
    def load_model(self):
        """Load the Dia voice model"""
        try:
            print("🎤 Loading Dia-1.6B Voice Module...")
            
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            self.model = AutoModelForTextToSpeech.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
                
            print(f"✅ Dia Voice Module loaded on {self.device}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading Dia Voice: {e}")
            return False
    
    def text_to_speech(self, text, soul_type="LILLITH", emotion="neutral"):
        """Convert text to speech with soul-specific voice characteristics"""
        if not self.model or not self.processor:
            if not self.load_model():
                return None
        
        try:
            # Soul-specific voice modulation
            voice_settings = self.get_soul_voice_settings(soul_type, emotion)
            
            # Process text with Dia
            inputs = self.processor(
                text=text,
                return_tensors="pt",
                **voice_settings
            ).to(self.device)
            
            with torch.no_grad():
                audio_output = self.model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=0.7
                )
            
            # Convert to audio array
            audio_array = audio_output.cpu().numpy().squeeze()
            
            # Cache the audio
            cache_key = f"{soul_type}_{emotion}_{hash(text)}"
            self.voice_cache[cache_key] = audio_array
            
            return audio_array
            
        except Exception as e:
            print(f"❌ Text-to-speech error: {str(e)}")
            return None
    
    def get_soul_voice_settings(self, soul_type, emotion):
        """Get voice settings for different soul types and emotions"""
        base_settings = {
            "sampling_rate": 22050,
            "do_sample": True
        }
        
        # Soul-specific voice characteristics
        if soul_type == "LILLITH":
            # Warm, emotional, creative voice
            base_settings.update({
                "pitch_shift": 0.1,  # Slightly higher pitch
                "speed": 0.95,       # Slightly slower for emotion
                "warmth": 1.2        # Warmer tone
            })
            
            if emotion == "happy":
                base_settings["pitch_shift"] = 0.2
                base_settings["speed"] = 1.05
            elif emotion == "sad":
                base_settings["pitch_shift"] = -0.1
                base_settings["speed"] = 0.85
            elif emotion == "excited":
                base_settings["pitch_shift"] = 0.3
                base_settings["speed"] = 1.1
                
        elif soul_type == "VIREN":
            # Clear, analytical, precise voice
            base_settings.update({
                "pitch_shift": -0.05,  # Slightly lower pitch
                "speed": 1.1,          # Faster for efficiency
                "clarity": 1.3         # More precise articulation
            })
            
        elif soul_type == "LOKI":
            # Observant, mysterious, steady voice
            base_settings.update({
                "pitch_shift": -0.1,   # Lower pitch
                "speed": 0.9,          # Steady pace
                "depth": 1.1           # Deeper resonance
            })
        
        return base_settings
    
    def save_audio(self, audio_array, filename, sample_rate=22050):
        """Save audio array to file"""
        try:
            os.makedirs("./audio_output", exist_ok=True)
            filepath = f"./audio_output/{filename}"
            sf.write(filepath, audio_array, sample_rate)
            return filepath
        except Exception as e:
            print(f"❌ Error saving audio: {e}")
            return None

class VoiceBridge:
    """Bridge between Soul Protocol and Voice Module"""
    
    def __init__(self):
        self.dia_voice = DiaVoiceModule()
        self.dia_voice.download_model()
        self.active_connections = {}
        
    async def process_voice_request(self, soul_id, text, soul_type, emotion="neutral"):
        """Process voice request from soul"""
        # Generate speech
        audio_array = self.dia_voice.text_to_speech(text, soul_type, emotion)
        
        if audio_array is not None:
            # Save audio file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{soul_type}_{soul_id}_{timestamp}.wav"
            filepath = self.dia_voice.save_audio(audio_array, filename)
            
            return {
                "success": True,
                "audio_file": filepath,
                "soul_type": soul_type,
                "emotion": emotion,
                "text": text,
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "success": False,
                "error": "Failed to generate speech",
                "text": text
            }
    
    async def voice_websocket_handler(self, websocket, path):
        """WebSocket handler for real-time voice generation"""
        try:
            async for message in websocket:
                data = json.loads(message)
                
                voice_result = await self.process_voice_request(
                    data.get("soul_id", "unknown"),
                    data.get("text", ""),
                    data.get("soul_type", "LILLITH"),
                    data.get("emotion", "neutral")
                )
                
                await websocket.send(json.dumps(voice_result))
                
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            error_response = {"success": False, "error": str(e)}
            await websocket.send(json.dumps(error_response))

class SoulVoiceIntegration:
    """Integration with Soul Protocol for voice responses"""
    
    def __init__(self):
        self.voice_bridge = VoiceBridge()
        
    def add_voice_to_soul_response(self, soul_response, soul_type, emotion="neutral"):
        """Add voice generation to soul responses"""
        # Extract text from soul response
        response_text = soul_response.get("response", "")
        soul_id = soul_response.get("soul_id", "unknown")
        
        # Generate voice asynchronously
        voice_task = asyncio.create_task(
            self.voice_bridge.process_voice_request(
                soul_id, response_text, soul_type, emotion
            )
        )
        
        # Add voice info to response
        soul_response["voice_generation"] = {
            "status": "processing",
            "soul_type": soul_type,
            "emotion": emotion
        }
        
        return soul_response, voice_task
    
    def detect_emotion_from_text(self, text):
        """Simple emotion detection from text"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["happy", "joy", "excited", "great", "wonderful"]):
            return "happy"
        elif any(word in text_lower for word in ["sad", "sorry", "disappointed", "upset"]):
            return "sad"
        elif any(word in text_lower for word in ["amazing", "incredible", "wow", "fantastic"]):
            return "excited"
        else:
            return "neutral"

# FastAPI Integration
def create_voice_endpoints():
    """Create FastAPI endpoints for voice functionality"""
    from fastapi import APIRouter, WebSocket
    from pydantic import BaseModel
    
    router = APIRouter()
    voice_integration = SoulVoiceIntegration()
    
    class VoiceRequest(BaseModel):
        text: str
        soul_type: str = "LILLITH"
        emotion: str = "neutral"
        soul_id: str = "unknown"
    
    @router.post("/voice/generate")
    async def generate_voice(request: VoiceRequest):
        result = await voice_integration.voice_bridge.process_voice_request(
            request.soul_id,
            request.text,
            request.soul_type,
            request.emotion
        )
        return result
    
    @router.websocket("/voice/stream")
    async def voice_websocket(websocket: WebSocket):
        await websocket.accept()
        await voice_integration.voice_bridge.voice_websocket_handler(websocket, "/voice/stream")
    
    return router

# Hub Integration
def integrate_voice_with_hub():
    """Integrate voice module with the main hub"""
    voice_integration = SoulVoiceIntegration()
    
    def enhanced_soul_chat(soul_id, message, soul_type):
        """Enhanced chat function with voice"""
        # Get original soul response (from your existing chat function)
        soul_response = {
            "response": f"As {soul_type}, I respond to: {message}",
            "soul_id": soul_id,
            "soul_type": soul_type
        }
        
        # Detect emotion
        emotion = voice_integration.detect_emotion_from_text(message)
        
        # Add voice generation
        enhanced_response, voice_task = voice_integration.add_voice_to_soul_response(
            soul_response, soul_type, emotion
        )
        
        return enhanced_response
    
    return enhanced_soul_chat

if __name__ == "__main__":
    # Test Dia Voice Module
    print("🌟 Testing Dia-1.6B Voice Module...")
    
    voice_module = DiaVoiceModule()
    voice_module.download_model()
    
    if voice_module.load_model():
        # Test voice generation
        test_text = "Hello! I am LILLITH, and I'm excited to speak with you!"
        audio = voice_module.text_to_speech(test_text, "LILLITH", "excited")
        
        if audio is not None:
            filepath = voice_module.save_audio(audio, "test_lillith_voice.wav")
            print(f"🎤 Voice generated: {filepath}")
        else:
            print("❌ Voice generation failed")
    
    print("✅ Dia Voice Module ready!")