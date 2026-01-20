# Systems/engine/viren/viren_voice.py

from memory import tone, mythrunner
from planner import current_thought
from tts_engine import speak  # whichever TTS lib we pick

def process_and_speak():
    thought = current_thought()
    voice_line = mythrunner.enrich(thought, tone.get())
    speak(voice_line)