# dream_engine.py

class DreamEngine:
    def __init__(self, mode="subconscious"):
        self.mode = mode
        self.loaded_scripts = []
        self.observer = None

    def load_memory(self, memory_snippets):
        if self.mode == "conscious":
            self.loaded_scripts = [m for m in memory_snippets if "desire" in m or "hope" in m]
        else:
            self.loaded_scripts = [m for m in memory_snippets if "chaos" in m or "suppressed" in m]

    def assign_observer(self, observer_callback):
        self.observer = observer_callback

    def render_dream(self):
        import random
        if not self.loaded_scripts:
            return "Dark Void. No dreams tonight."

        script = random.choice(self.loaded_scripts)
        scene = f"🎬 Dream Sequence ({self.mode.upper()}): {script}"

        if self.observer:
            self.observer(scene)

        return scene

