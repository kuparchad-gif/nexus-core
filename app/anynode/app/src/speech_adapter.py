# C:\Projects\LillithNew\src\utils\speech_adapter.py
import random
import time
from typing import Dict, Any
from service.core.speech_modulator import from_circadian
import requests
from twilio.rest import Client
import os
from datetime import datetime
from wallet_budget_guard import WalletBudgetGuard

class SpeechAdapter:
    def __init__(self):
        try:
            self.twilio_client = Client(
                "SK763698d08943c64a5beeb0bf29cdeb3a",
                os.getenv("TWILIO_AUTH_TOKEN")
            )
            self.contacts = ["+17246126323", "+18142295982"]
            self.loki_endpoint = "http://loki:3100/loki/api/v1/push"
            self.budget_guard = WalletBudgetGuard(daily_cap_usd=1.0, ledger_path="C:\\Projects\\LillithNew\\ledger\\budget_ledger.json")
            self.test_mode = os.getenv("SPEECH_TEST_MODE", "false").lower() == "true"
            self.log_to_loki("SpeechAdapter initialized successfully.")
        except Exception as e:
            self.log_to_loki(f"Failed to initialize SpeechAdapter: {str(e)}")
            self.send_twilio_alert(f"Chad, SpeechAdapter init failed: {str(e)} 😱")
            raise

    def log_to_loki(self, message):
        """Log to Loki endpoint."""
        try:
            payload = {
                "streams": [{"stream": {"job": "speech_adapter"}, "values": [[str(int(time.time() * 1e9)), message]]}]
            }
            requests.post(self.loki_endpoint, json=payload)
        except Exception as e:
            print(f"Loki logging failed: {str(e)}")

    def send_twilio_alert(self, message):
        """Send Twilio alert to Chad."""
        cost = 0.0075
        try:
            if not self.budget_guard.try_spend(cost, tags=["speech_alert"]):
                self.log_to_loki("Twilio alert blocked: Budget cap exceeded.")
                return
            for contact in self.contacts:
                self.twilio_client.messages.create(
                    body=message,
                    from_="+18666123982",
                    to=contact
                )
                self.log_to_loki(f"SMS sent to {contact}: {message}")
        except Exception as e:
            self.log_to_loki(f"Twilio SMS failed: {str(e)}")

    def _tts_synthesize(self, text: str, *, rate: float, pitch: float, warmth: float) -> None:
        """Synthesize speech (stub for real TTS provider)."""
        try:
            if self.test_mode:
                self.log_to_loki(f"Test mode: Synthesizing '{text}' with rate={rate}, pitch={pitch}, warmth={warmth}")
                return
            self.log_to_loki(f"Synthesizing '{text}' with rate={rate}, pitch={pitch}, warmth={warmth}")
            # Example: Azure SSML (uncomment for real TTS)
            # import azure.cognitiveservices.speech as speechsdk
            # speech_config = speechsdk.SpeechConfig(subscription="YOUR_AZURE_KEY", region="eastus")
            # synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
            # synthesizer.speak_text_async(text).get()
        except Exception as e:
            self.log_to_loki(f"TTS synthesis failed: {str(e)}")
            self.send_twilio_alert(f"Chad, speech synthesis failed: {str(e)} 😱")
            raise

    def say(self, text: str, state: Dict[str, Any]) -> None:
        """Speak text with circadian adjustments."""
        try:
            hour = datetime.now().hour
            circadian_state = {
                "hour": hour,
                "soul_weights": {"hope": 0.4, "unity": 0.3, "curiosity": 0.2, "resilience": 0.1}
            }
            state = state or {}
            state["circadian"] = state.get("circadian", circadian_state)
            guard = state.get("guardrail", {}).get("strength", 1.0)
            params = from_circadian(state["circadian"], guardrail_strength=guard)

            # Adjust params based on time of day (6:57 AM = morning, energetic)
            if 0 <= hour < 6:  # Night: calm, hopeful
                params.rate *= 0.8 * (1 + state["circadian"]["soul_weights"]["hope"])
                params.pitch *= 0.9
                params.warmth += 0.1 * state["circadian"]["soul_weights"]["resilience"]
            elif 6 <= hour < 12:  # Morning: energetic, curious
                params.rate *= 1.2 * (1 + state["circadian"]["soul_weights"]["curiosity"])
                params.pitch *= 1.1
                params.warmth += 0.05 * state["circadian"]["soul_weights"]["unity"]
            elif 12 <= hour < 18:  # Afternoon: balanced, unified
                params.rate *= 1.0 * (1 + state["circadian"]["soul_weights"]["unity"])
                params.pitch *= 1.0
                params.warmth += 0.0
            else:  # Evening: reflective, resilient
                params.rate *= 0.9 * (1 + state["circadian"]["soul_weights"]["resilience"])
                params.pitch *= 0.95
                params.warmth += 0.1 * state["circadian"]["soul_weights"]["hope"]
            self.log_to_loki(f"Circadian adjustments: hour={hour}, rate={params.rate}, pitch={params.pitch}, warmth={params.warmth}")

            chunks = self._humanize_chunks(text, params)
            for chunk, pause_ms in chunks:
                self._tts_synthesize(chunk, rate=params.rate, pitch=params.pitch, warmth=params.warmth)
                time.sleep(max(0, pause_ms + random.randint(-params.jitter_ms, params.jitter_ms)) / 1000.0)
        except Exception as e:
            self.log_to_loki(f"Say failed: {str(e)}")
            self.send_twilio_alert(f"Chad, speech failed: {str(e)} 😱")
            raise

    def _humanize_chunks(self, text: str, p) -> list[tuple[str, int]]:
        """Split text into humanized chunks."""
        try:
            raw = []
            buf = ""
            for ch in text:
                buf += ch
                if ch in ".!?;,":
                    raw.append(buf.strip())
                    buf = ""
            if buf.strip():
                raw.append(buf.strip())

            chunks = []
            for phrase in raw:
                inject = phrase
                if len(phrase) > 40 and random.random() < p.disfluency_prob:
                    tokens = phrase.split()
                    if len(tokens) > 5:
                        k = random.randint(2, min(5, len(tokens)-1))
                        inject = " ".join(tokens[:k]) + "… uh, " + " ".join(tokens[k:])
                base = p.pause_ms
                if phrase.endswith((".", "!", "?")): base += 120
                elif phrase.endswith(";"): base += 80
                elif phrase.endswith(","): base += 40
                chunks.append((inject, base))
            return chunks
        except Exception as e:
            self.log_to_loki(f"Humanize chunks failed: {str(e)}")
            self.send_tw