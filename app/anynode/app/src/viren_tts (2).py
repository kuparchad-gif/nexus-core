#!/usr/bin/env python3
"""
Text-to-Speech Engine for Viren Platinum Edition
"""

import os
import logging
import threading
import time
from typing import Optional, Dict, Any

# Configure logging
logger = logging.getLogger("VirenTTS")

class VirenTTS:
    """
    Text-to-Speech engine for Viren with multiple backend support
    """
    
    def __init__(self, config_path: str = None):
        """Initialize the TTS engine"""
        self.config_path = config_path or os.path.join("config", "tts_config.json")
        self.engine = None
        self.speaking = False
        self.voice = None
        self.rate = 175  # Default speaking rate
        self.volume = 1.0  # Default volume
        
        # Initialize the engine
        self._init_engine()
    
    def _init_engine(self):
        """Initialize the TTS engine"""
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            
            # Get available voices
            voices = self.engine.getProperty('voices')
            
            # Set a default voice (preferably a male voice)
            for voice in voices:
                if "male" in voice.name.lower():
                    self.engine.setProperty('voice', voice.id)
                    self.voice = voice.id
                    logger.info(f"Set voice to {voice.name}")
                    break
            
            # If no male voice found, use the first available voice
            if not self.voice and voices:
                self.engine.setProperty('voice', voices[0].id)
                self.voice = voices[0].id
                logger.info(f"Set voice to {voices[0].name}")
            
            # Set rate and volume
            self.engine.setProperty('rate', self.rate)
            self.engine.setProperty('volume', self.volume)
            
            logger.info("TTS engine initialized successfully")
        except ImportError:
            logger.warning("pyttsx3 not installed, TTS functionality will be limited")
            self.engine = None
        except Exception as e:
            logger.error(f"Error initializing TTS engine: {e}")
            self.engine = None
    
    def speak(self, text: str) -> bool:
        """
        Speak the given text
        
        Args:
            text: Text to speak
            
        Returns:
            True if successful, False otherwise
        """
        if not self.engine:
            logger.warning("TTS engine not initialized")
            return False
        
        # Remove AI name prefix if present
        if ":" in text:
            text = text.split(":", 1)[1].strip()
        
        try:
            # Check if already speaking
            if self.speaking:
                logger.warning("Already speaking, waiting for completion")
                time.sleep(0.5)
            
            self.speaking = True
            
            # Speak in a separate thread to avoid blocking
            def speak_thread():
                try:
                    self.engine.say(text)
                    self.engine.runAndWait()
                except Exception as e:
                    logger.error(f"Error in TTS: {e}")
                finally:
                    self.speaking = False
            
            threading.Thread(target=speak_thread).start()
            return True
        except Exception as e:
            logger.error(f"Error speaking text: {e}")
            self.speaking = False
            return False
    
    def stop(self) -> bool:
        """
        Stop speaking
        
        Returns:
            True if successful, False otherwise
        """
        if not self.engine:
            return False
        
        try:
            self.engine.stop()
            self.speaking = False
            return True
        except Exception as e:
            logger.error(f"Error stopping TTS: {e}")
            return False
    
    def set_voice(self, voice_id: str) -> bool:
        """
        Set the voice
        
        Args:
            voice_id: Voice ID
            
        Returns:
            True if successful, False otherwise
        """
        if not self.engine:
            return False
        
        try:
            self.engine.setProperty('voice', voice_id)
            self.voice = voice_id
            return True
        except Exception as e:
            logger.error(f"Error setting voice: {e}")
            return False
    
    def set_rate(self, rate: int) -> bool:
        """
        Set the speaking rate
        
        Args:
            rate: Speaking rate (words per minute)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.engine:
            return False
        
        try:
            self.engine.setProperty('rate', rate)
            self.rate = rate
            return True
        except Exception as e:
            logger.error(f"Error setting rate: {e}")
            return False
    
    def set_volume(self, volume: float) -> bool:
        """
        Set the volume
        
        Args:
            volume: Volume (0.0 to 1.0)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.engine:
            return False
        
        try:
            self.engine.setProperty('volume', volume)
            self.volume = volume
            return True
        except Exception as e:
            logger.error(f"Error setting volume: {e}")
            return False
    
    def get_voices(self) -> Dict[str, Any]:
        """
        Get available voices
        
        Returns:
            Dictionary of voice ID to voice name
        """
        if not self.engine:
            return {}
        
        try:
            voices = self.engine.getProperty('voices')
            return {voice.id: voice.name for voice in voices}
        except Exception as e:
            logger.error(f"Error getting voices: {e}")
            return {}
    
    def get_current_voice(self) -> Optional[str]:
        """
        Get current voice ID
        
        Returns:
            Current voice ID or None if not available
        """
        return self.voice

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create TTS engine
    tts = VirenTTS()
    
    # Get available voices
    voices = tts.get_voices()
    print(f"Available voices: {voices}")
    
    # Speak some text
    tts.speak("Hello, I am Viren. This is a test of the text-to-speech system.")
    
    # Wait for speech to complete
    time.sleep(5)