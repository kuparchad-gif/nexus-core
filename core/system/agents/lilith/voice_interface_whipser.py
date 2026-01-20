import whisper
from speech_recognition import Microphone, Recognizer

class LillithVoiceInterface:
    def __init__(self):
        self.model = whisper.load_model("base")
        self.recognizer = Recognizer()
        
    async def listen_for_command(self):
        """Listen for voice commands and route to appropriate domains"""
        with Microphone() as source:
            print("🎤 Lillith listening...")
            audio = self.recognizer.listen(source, timeout=5)
            
        try:
            # Save audio and transcribe
            with open("temp_command.wav", "wb") as f:
                f.write(audio.get_wav_data())
                
            result = self.model.transcribe("temp_command.wav")
            command = result["text"].lower()
            
            # Route to appropriate domain
            if "docker" in command or "error" in command:
                return await self.handle_troubleshooting(command)
            elif "stock" in command or "market" in command:
                return await self.handle_stocks(command) 
            elif "account" in command or "tax" in command:
                return await self.handle_accounting(command)
            elif any(word in command for word in ["unity", "spirit", "divine"]):
                return await self.handle_spirituality(command)
                
        except Exception as e:
            return f"Voice processing error: {e}"