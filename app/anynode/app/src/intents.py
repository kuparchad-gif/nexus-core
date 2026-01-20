# language_processing/app/alexa_bridge/app/intents.py
# Intent handlers for the Alexa Bridge.
import os
import re
import hashlib
from typing import Any, Dict, Optional

from ask_sdk_core.dispatch_components import AbstractRequestHandler, AbstractExceptionHandler
from ask_sdk_core.handler_input import HandlerInput
from ask_sdk_core.utils import is_intent_name, is_request_type
from ask_sdk_model import Response
from ask_sdk_model.interfaces.audioplayer import StopDirective

from .lilith_client import ask_lilith

MAX_SPEECH_CHARS = 7500  # keep under Alexa's ~8000 char limit

def _sanitize(text: str) -> str:
    # Basic sanitation to keep speech concise
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_SPEECH_CHARS]

def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

class LaunchRequestHandler(AbstractRequestHandler):
    def can_handle(self, handler_input: HandlerInput) -> bool:
        return is_request_type("LaunchRequest")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        speak_output = "Lilith is live. What do you want to ask?"
        reprompt = "Go ahead—ask Lilith anything."
        return handler_input.response_builder.speak(speak_output).ask(reprompt).response

class AskLilithIntentHandler(AbstractRequestHandler):
    def __init__(self, safe_mode: bool):
        self.safe_mode = safe_mode

    def can_handle(self, handler_input: HandlerInput) -> bool:
        return is_intent_name("AskLilithIntent")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        req = handler_input.request_envelope
        query = ""
        try:
            query = req.request.intent.slots["query"].value or ""
        except Exception:
            query = ""

        # Pull deviceId and apiEndpoint for optional room awareness or features
        device_id = None
        api_endpoint = None
        user_id = None
        try:
            sys = req.context.system
            device_id = getattr(sys.device, "device_id", None)
            api_endpoint = getattr(sys, "api_endpoint", None)
            user_id = getattr(sys.user, "user_id", None)
        except Exception:
            pass

        # Account linking token (if configured in Alexa console)
        access_token = None
        try:
            access_token = req.context.system.user.access_token
        except Exception:
            access_token = None

        # Call Lilith (UCE Router) with safe-mode system prompt
        answer_text = ask_lilith(
            query=query or "Say hello to the user.",
            device_id=device_id,
            room_hint=None,  # resolved by room_map.json in the client call
            user_hash=_sha(user_id or "anon"),
            safe_mode=self.safe_mode,
            access_token=access_token,
        )

        speak_output = _sanitize(answer_text or "Sorry, I couldn't get a response from Lilith yet.")
        # Keep session open for a follow-up
        return handler_input.response_builder.speak(speak_output).ask("Anything else?").response

class HelpIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input: HandlerInput) -> bool:
        return is_intent_name("AMAZON.HelpIntent")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        speak_output = "You can say, 'Ask Lilith' and then your question. For deeper features, link your account in the Alexa app."
        return handler_input.response_builder.speak(speak_output).ask("What would you like to try?").response

class CancelOrStopIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input: HandlerInput) -> bool:
        return is_intent_name("AMAZON.CancelIntent")(handler_input) or is_intent_name("AMAZON.StopIntent")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        return handler_input.response_builder.add_directive(StopDirective()).speak("Goodbye.").set_should_end_session(True).response

class FallbackIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input: HandlerInput) -> bool:
        return is_intent_name("AMAZON.FallbackIntent")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        return handler_input.response_builder.speak("I didn't catch that. Try, 'Ask Lilith' and your question.").ask("What would you like to ask?").response

class CatchAllExceptionHandler(AbstractExceptionHandler):
    def can_handle(self, handler_input, exception) -> bool:
        return True

    def handle(self, handler_input, exception) -> Response:
        return handler_input.response_builder.speak("Something went sideways. Please try again.").ask("What would you like to ask?").response

def register_intents(sb, safe_mode: bool = True):
    sb.add_request_handler(LaunchRequestHandler())
    sb.add_request_handler(AskLilithIntentHandler(safe_mode=safe_mode))
    sb.add_request_handler(HelpIntentHandler())
    sb.add_request_handler(CancelOrStopIntentHandler())
    sb.add_request_handler(FallbackIntentHandler())
    sb.add_exception_handler(CatchAllExceptionHandler())
