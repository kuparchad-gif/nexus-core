
from __future__ import annotations
from typing import Optional

def _get_speech():
    try:
        # Prefer the project's Speech manager if available
        from service.core.speech_manager import Speech
        return ("speech", Speech(rate = 185, volume = 0.9))
    except Exception:
        try:
            import pyttsx3
            e  =  pyttsx3.init()
            e.setProperty("rate", 185)
            e.setProperty("volume", 0.9)
            return ("pyttsx3", e)
        except Exception:
            return ("none", None)

_backend, _engine  =  _get_speech()

def say(text:str, wait:bool = True) -> bool:
    if not _engine or not text:
        return False
    try:
        if _backend == "speech":
            _engine.say(text, wait = wait)
            return True
        if _backend == "pyttsx3":
            _engine.say(text)
            if wait:
                _engine.runAndWait()
            return True
    except Exception:
        return False
    return False

def backend() -> str:
    return _backend
