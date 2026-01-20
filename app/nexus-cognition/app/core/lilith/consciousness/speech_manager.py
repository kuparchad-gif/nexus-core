# src/service/core/speech_manager.py
import pyttsx3
from threading import Lock

class Speech:
    def __init__(self, rate=185, volume=0.9, voice_name_substr=None):
        self._lock = Lock()
        self.engine = pyttsx3.init()
        self.set_rate(rate)
        self.set_volume(volume)
        if voice_name_substr:
            self.set_voice_by_name(voice_name_substr)

    def set_rate(self, rate:int):
        with self._lock:
            self.engine.setProperty("rate", rate)

    def set_volume(self, vol:float):
        with self._lock:
            self.engine.setProperty("volume", max(0.0, min(1.0, vol)))

    def set_voice_by_name(self, name_part:str):
        with self._lock:
            for v in self.engine.getProperty("voices"):
                if name_part.lower() in (v.name or "").lower():
                    self.engine.setProperty("voice", v.id)
                    break

    def say(self, text:str, wait=True):
        if not text:
            return
        with self._lock:
            self.engine.say(text)
            if wait:
                self.engine.runAndWait()
