#!/usr/bin/env python3
"""
Speech-to-Text Engine for Viren Platinum Edition
"""

import os
import logging
from typing import Optional, Dict, Any

# Configure logging
logger = logging.getLogger("VirenSTT")

class VirenSTT:
    """
    Speech-to-Text engine for Viren with multiple backend support
    """
    
    def __init__(self, config_path: str = None):
        """Initialize the STT engine"""
        self.config_path = config_path or os.path.join("config", "stt_config.json")
        self.recognizer = None
        self.engine = "google"  # Default engine
        
        # Initialize the recognizer
        self._init_recognizer()
    
    def _init_recognizer(self):
        """Initialize the speech recognizer"""
        try:
            import speech_recognition as sr
            self.recognizer = sr.Recognizer()
            logger.info("Speech recognizer initialized successfully")
        except ImportError:
            logger.warning("speech_recognition not installed, STT functionality will be limited")
            self.recognizer = None
        except Exception as e:
            logger.error(f"Error initializing speech recognizer: {e}")
            self.recognizer = None
    
    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe audio file to text
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        if not self.recognizer:
            logger.warning("Speech recognizer not initialized")
            return "Speech recognition not available"
        
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return "Audio file not found"
        
        try:
            import speech_recognition as sr
            
            with sr.AudioFile(audio_path) as source:
                audio_data = self.recognizer.record(source)
                
                if self.engine == "google":
                    text = self.recognizer.recognize_google(audio_data)
                elif self.engine == "sphinx":
                    text = self.recognizer.recognize_sphinx(audio_data)
                elif self.engine == "whisper":
                    text = self._recognize_whisper(audio_data)
                else:
                    text = self.recognizer.recognize_google(audio_data)
                
                logger.info(f"Transcribed: {text}")
                return text
        except sr.UnknownValueError:
            logger.warning("Speech recognition could not understand audio")
            return "Could not understand audio"
        except sr.RequestError as e:
            logger.error(f"Speech recognition service error: {e}")
            return f"Speech recognition service error: {e}"
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return f"Error: {str(e)}"
    
    def _recognize_whisper(self, audio_data) -> str:
        """
        Recognize speech using OpenAI Whisper
        
        Args:
            audio_data: Audio data
            
        Returns:
            Transcribed text
        """
        try:
            import whisper
            
            # Save audio data to temporary file
            temp_file = "temp_audio.wav"
            with open(temp_file, "wb") as f:
                f.write(audio_data.get_wav_data())
            
            # Load model and transcribe
            model = whisper.load_model("base")
            result = model.transcribe(temp_file)
            
            # Clean up
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            return result["text"]
        except ImportError:
            logger.warning("whisper not installed, falling back to Google")
            return self.recognizer.recognize_google(audio_data)
        except Exception as e:
            logger.error(f"Error using Whisper: {e}")
            return self.recognizer.recognize_google(audio_data)
    
    def set_engine(self, engine: str) -> bool:
        """
        Set the speech recognition engine
        
        Args:
            engine: Engine name (google, sphinx, whisper)
            
        Returns:
            True if successful, False otherwise
        """
        valid_engines = ["google", "sphinx", "whisper"]
        if engine not in valid_engines:
            logger.warning(f"Invalid engine: {engine}. Valid options are: {valid_engines}")
            return False
        
        # Check if engine is available
        if engine == "sphinx":
            try:
                import speech_recognition as sr
                self.recognizer.recognize_sphinx
                self.engine = engine
                return True
            except (ImportError, AttributeError):
                logger.warning("sphinx not available, falling back to google")
                self.engine = "google"
                return False
        elif engine == "whisper":
            try:
                import whisper
                self.engine = engine
                return True
            except ImportError:
                logger.warning("whisper not available, falling back to google")
                self.engine = "google"
                return False
        else:
            self.engine = engine
            return True
    
    def get_engine(self) -> str:
        """
        Get current speech recognition engine
        
        Returns:
            Engine name
        """
        return self.engine
    
    def listen_from_microphone(self, timeout: int = 5) -> Optional[str]:
        """
        Listen for speech from microphone
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Transcribed text or None if failed
        """
        if not self.recognizer:
            logger.warning("Speech recognizer not initialized")
            return None
        
        try:
            import speech_recognition as sr
            
            with sr.Microphone() as source:
                logger.info("Listening...")
                self.recognizer.adjust_for_ambient_noise(source)
                audio = self.recognizer.listen(source, timeout=timeout)
                
                if self.engine == "google":
                    text = self.recognizer.recognize_google(audio)
                elif self.engine == "sphinx":
                    text = self.recognizer.recognize_sphinx(audio)
                elif self.engine == "whisper":
                    text = self._recognize_whisper(audio)
                else:
                    text = self.recognizer.recognize_google(audio)
                
                logger.info(f"Transcribed: {text}")
                return text
        except sr.UnknownValueError:
            logger.warning("Speech recognition could not understand audio")
            return None
        except sr.RequestError as e:
            logger.error(f"Speech recognition service error: {e}")
            return None
        except Exception as e:
            logger.error(f"Error listening from microphone: {e}")
            return None

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create STT engine
    stt = VirenSTT()
    
    # Get current engine
    engine = stt.get_engine()
    print(f"Current engine: {engine}")
    
    # Transcribe audio file if provided
    import sys
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        text = stt.transcribe(audio_path)
        print(f"Transcribed text: {text}")
    else:
        print("No audio file provided. Usage: python viren_stt.py <audio_file>")
        
        # Try listening from microphone
        print("Listening from microphone for 5 seconds...")
        text = stt.listen_from_microphone(5)
        if text:
            print(f"Transcribed text: {text}")
        else:
            print("No speech detected or error occurred")