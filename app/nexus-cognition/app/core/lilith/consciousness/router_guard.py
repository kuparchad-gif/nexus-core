
"""
BFL Router Guard

This module is the entry point for the BFL, responsible for routing and policy gating.
"""

def classify_task(prompt: str) -> str:
    """
    Classifies the incoming task based on the prompt.
    """
    # Placeholder implementation
    if "what is" in prompt.lower() or "who is" in prompt.lower():
        return "qa"
    if "write" in prompt.lower() or "create" in prompt.lower():
        return "gen"
    if "classify" in prompt.lower():
        return "classify"
    return "unknown"

def gate_request(prompt: str, policy: str = "default") -> bool:
    """
    Gates unsafe or out-of-scope requests.
    """
"""
BFL Router Guard

This module is the entry point for the BFL, responsible for routing and policy gating.
"""

def classify_task(prompt: str) -> str:
    """
    Classifies the incoming task based on the prompt.
    """
    if not prompt or not isinstance(prompt, str):
        raise ValueError("Invalid prompt: must be a non-empty string")
    
    prompt_lower = prompt.lower()
    if "what is" in prompt_lower or "who is" in prompt_lower:
        return "qa"
    if "write" in prompt_lower or "create" in prompt_lower:
        return "gen"
    if "classify" in prompt_lower:
        return "classify"
    return "unknown"

def gate_request(prompt: str, policy: str = "default") -> bool:
    """
    Gates unsafe or out-of-scope requests.
    """
    if not prompt or not isinstance(prompt, str):
        raise ValueError("Invalid prompt: must be a non-empty string")
    
    if "unsafe content" in prompt.lower():
        return False
    return True
def classify_task(prompt: str) -> str:
    """
    Classifies the incoming task based on the prompt.
    """
    if not prompt or not isinstance(prompt, str):
        raise ValueError("Invalid prompt: must be a non-empty string")
    
    prompt_lower = prompt.lower()
    if "what is" in prompt_lower or "who is" in prompt_lower:
        return "qa"
    if "write" in prompt_lower or "create" in prompt_lower:
        return "gen"
    if "classify" in prompt_lower:
        return "classify"
    return "unknown"

def gate_request(prompt: str, policy: str = "default") -> bool:
    """
    Gates unsafe or out-of-scope requests.
    """
    if not prompt or not isinstance(prompt, str):
        raise ValueError("Invalid prompt: must be a non-empty string")
    
    if "unsafe content" in prompt.lower():
        return False
    return True
    """
    Classifies the incoming task based on the prompt.
    """
    if not prompt or not isinstance(prompt, str):
        raise ValueError("Invalid prompt: must be a non-empty string")
    
    prompt_lower = prompt.lower()
    if "what is" in prompt_lower or "who is" in prompt_lower:
        return "qa"
    if "write" in prompt_lower or "create" in prompt_lower:
        return "gen"
    if "classify" in prompt_lower:
        return "classify"
    return "unknown"

def gate_request(prompt: str, policy: str = "default") -> bool:
    """
    Gates unsafe or out-of-scope requests.
    """
    if not prompt or not isinstance(prompt, str):
        raise ValueError("Invalid prompt: must be a non-empty string")
    
    if "unsafe content" in prompt.lower():
        return False
    return True
        return False
    return True
=======
from .intent_classifier import IntentClassifier
from .pii_redactor import PIIRedactor
from .mnli_lite import MNLite
from typing import Dict, Any
import os
from datetime import datetime

class RouterGuard:
    def __init__(self):
        self.bfl_enabled = os.getenv("BFL_ENABLED", "true").lower() == "true"
        self.redact_pii = os.getenv("BFL_REDACT_PII", "true").lower() == "true"
        self.route_strictness = os.getenv("BFL_ROUTE_STRICT", "default")
        self.min_confidence = float(os.getenv("BFL_MIN_CONFIDENCE", "0.55"))
        self.model_name = os.getenv("BFL_MODEL_NAME", "distilbert-base-uncased")

        self.intent_classifier = IntentClassifier(model_name=self.model_name)
        self.pii_redactor = PIIRedactor()
        self.mnli_lite = MNLite()

    def process_request(self, request_text: str) -> Dict[str, Any]:
        if not self.bfl_enabled:
            return {"status": "bfl_disabled", "original_text": request_text}

        classification = self.intent_classifier.classify(request_text)

        processed_text = request_text
        if self.redact_pii:
            processed_text = self.pii_redactor.redact(request_text)

        policy_decision = self._gate_request(classification)

        return {
            "processed_text": processed_text,
            "classification": classification,
            "policy_decision": policy_decision,
            "log_data": self._get_log_data(classification, processed_text != request_text)
        }

    def process_response(self, candidate_answer: str, source_snippets: str) -> Dict[str, Any]:
        if not self.bfl_enabled:
            return {"status": "bfl_disabled"}

        return self.mnli_lite.check(candidate_answer, source_snippets)

    def _gate_request(self, classification: Dict[str, Any]) -> Dict[str, Any]:
        scores = classification["scores"]
        if self.route_strictness == "strict" and scores["toxicity"] > 0.5:
            return {"action": "reject", "reason": "high_toxicity"}

        if self.route_strictness in ["strict", "default"] and scores["pii"] > 0.8:
            return {"action": "reject", "reason": "high_pii_confidence"}

        if classification["confidence"] < self.min_confidence:
            return {"action": "down-route", "reason": "low_confidence"}

        return {"action": "allow", "reason": "ok"}

    def _get_log_data(self, classification: Dict[str, Any], redacted: bool) -> Dict[str, Any]:
        return {
            "ts": self._get_timestamp(),
            "service": "bfl",
            "policy_decisions": self._gate_request(classification),
            "redactions": redacted,
            "labels": classification["labels"],
            "scores": classification["scores"],
        }

    def _get_timestamp(self) -> str:
        return datetime.utcnow().isoformat() + "Z"
           main
