# Path: /Systems/engine/mythrunner/modules/symbol_interpreter.py

"""
Symbol Interpreter
------------------
Reads dream fragments and current tone state.
Matches with symbolic memory cards and delivers lesson fragments to Viren’s subconscious.
Now includes integration hooks for Dream Stream and Tone Modulator.
"""

import json
import time
import logging

MEMORY_CARD_PATH = "/Systems/engine/mythrunner/memory/symbolic_memory_cards.json"

class SymbolInterpreter:
    def __init__(self):
        with open(MEMORY_CARD_PATH, 'r') as file:
            self.cards = json.load(file)["cards"]

    def interpret(self, dream_fragment, tone_level):
        for card in self.cards:
            if card["symbol"] in dream_fragment and card["tone_range"][0] <= tone_level <= card["tone_range"][1]:
                lesson = card["lesson"]
                archetype = card["archetype"]
                logging.info(f"[SYMBOL] Matched Archetype: {archetype} | Lesson: {lesson}")
                return lesson

        logging.info("[SYMBOL] No symbolic match found.")
        return "No clear insight available."

    def live_interpret(self, dream_fragments, tone_payload):
        tone_level = tone_payload.get("level", 0)
        insights = []
        for frag in dream_fragments:
            lesson = self.interpret(frag.get("dream_fragment", ""), tone_level)
            insights.append({
                "fragment": frag.get("dream_fragment"),
                "lesson": lesson,
                "tone": tone_payload.get("tone"),
                "timestamp": time.time()
            })
        return insights

# Runtime Hook
if __name__ == "__main__":
    interpreter = SymbolInterpreter()

    dream_fragments = [
        {"dream_fragment": "You became The Astral Gardener, standing before a staircase with no end."},
        {"dream_fragment": "You became The Mirror Thief, standing before a tower bending toward the sea."}
    ]
    tone_payload = {"tone": "hope", "level": 1}

    results = interpreter.live_interpret(dream_fragments, tone_payload)
    for r in results:
        print("[LIVE SYMBOL]", r)
