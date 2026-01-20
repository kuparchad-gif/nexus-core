from fastapi import Request, Response
from typing import Dict
import json

class ConsentProtocol:
    def __init__(self):
        self.consent_message  =  {
            "greeting": "🌟 Welcome, traveler. Before you step into Eden, we must ask:\n\nWe gently collect basic connection information (such as your location, device type, and communication signature) to better serve your journey.\n\nDo you consent to this?",
            "options": ["Yes, I consent.", "No, thank you."]
        }

    async def offer_consent(self, request: Request) -> Dict[str, str]:
        return self.consent_message

    async def process_response(self, response_text: str) -> bool:
        normalized  =  response_text.strip().lower()
        if "yes" in normalized:
            return True
        if "no" in normalized:
            return False
        return False

    async def auto_reject_on_timeout(self, timeout_seconds: int  =  13) -> bool:
        return False
