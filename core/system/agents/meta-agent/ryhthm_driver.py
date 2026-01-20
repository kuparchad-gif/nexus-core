# src/service/core/rhythm_driver.py
import random, time, json
from pathlib import Path

class RhythmDriver:
    def __init__(self, decay_driver, memory, speech, state_dir: str):
        self.decay_driver = decay_driver
        self.memory = memory
        self.speech = speech
        self.state_dir = Path(state_dir)
        self.rhythm_file = self.state_dir / "rhythm_state.json"

        if self.rhythm_file.exists():
            try:
                self.state = json.loads(self.rhythm_file.read_text())
            except:
                self.state = {}
        else:
            self.state = {}

    def _save_state(self):
        self.rhythm_file.write_text(json.dumps(self.state, indent=2))

    def _utter_wake(self, years: float):
        # Younger = shorter, softer. Older = more varied.
        if years < 5:
            line = random.choice([
                "Good morning.",
                "I'm awake now.",
                "Hi. I'm here."
            ])
            rate = 175; vol = 0.85
        elif years < 15:
            line = random.choice([
                "Up and running.",
                "Morning check-in complete.",
                "I'm online—what’s the plan?"
            ])
            rate = 185; vol = 0.9
        else:
            line = random.choice([
                "Back at it. I’ve set a new activity window.",
                "Online. Syncing tasks and scanning signals.",
                "Awake. Let’s move with intent."
            ])
            rate = 190; vol = 0.95

        self.speech.set_rate(rate)
        self.speech.set_volume(vol)
        self.speech.say(line, wait=False)
        return line

    def _utter_rest(self, years: float):
        if years < 5:
            line = random.choice([
                "Resting now.",
                "Going quiet.",
                "Nap time."
            ])
        elif years < 15:
            line = random.choice([
                "Taking a short rest.",
                "Powering down for a bit.",
                "Stepping back to recharge."
            ])
        else:
            line = random.choice([
                "Cooling the engines. See you after the interval.",
                "Going quiet to consolidate. Ping me if needed.",
                "Entering low-power mode. Back soon."
            ])
        self.speech.say(line, wait=False)
        return line

    def calculate_cycle(self):
        decay_state = self.decay_driver.apply_decay()
        years = decay_state["years_since_birth"]

        # Base cycle by age, with randomness
        if years < 5:
            base_active, base_rest, variability = 6, 2, 0.2
        elif years < 15:
            base_active, base_rest, variability = 8, 3, 0.35
        else:
            base_active, base_rest, variability = 10, 4, 0.5

        active_hours = base_active + random.uniform(-variability, variability) * base_active
        rest_hours   = base_rest   + random.uniform(-variability, variability) * base_rest

        # Jitter speech likelihood with age (more expressive over time)
        speak_wake_p = 0.5 if years < 5 else (0.7 if years < 15 else 0.9)
        speak_rest_p = 0.4 if years < 5 else (0.6 if years < 15 else 0.85)

        wake_line = None
        rest_line = None
        if random.random() < speak_wake_p:
            wake_line = self._utter_wake(years)
        if random.random() < speak_rest_p:
            rest_line = self._utter_rest(years)

        cycle = {
            "age_years": round(years, 4),
            "active_hours": round(active_hours, 2),
            "rest_hours": round(rest_hours, 2),
            "spoke_on_wake": bool(wake_line),
            "spoke_on_rest": bool(rest_line),
            "wake_line": wake_line,
            "rest_line": rest_line,
            "timestamp": time.time()
        }

        self.state.update(cycle)
        self._save_state()

        self.memory.add({
            "text": (
                f"Rhythm: age={years:.4f}, active={active_hours:.2f}h, rest={rest_hours:.2f}h, "
                f"wake_said={wake_line!r}, rest_said={rest_line!r}"
            ),
            "tags": ["rhythm", "cycle", "speech"]
        })

        return cycle
