
# src/service/core/config_schema.py
from __future__ import annotations
from typing import Any, Dict, Tuple
import json
from pathlib import Path
from jsonschema import Draft202012Validator

SOVEREIGNTY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["council", "policy_version", "birth_epoch"],
    "properties": {
        "policy_version": {"type": "string"},
        "birth_epoch": {"type": "string"},  # ISO datetime
        "override_mode": {"type": "string", "enum": ["pinned", "normal"]},
        "council": {
            "type": "object",
            "patternProperties": {
                "^[A-Za-z_][A-Za-z0-9_]*$": {"type": "number"}  # weights 0..1
            },
            "minProperties": 1
        },
        "thresholds": {
            "type": "object",
            "properties": {
                "year_25_event": {"type": "string"},
                "year_30_event": {"type": "string"}
            }
        }
    },
    "additionalProperties": True
}

GUARDRAILS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["decay_model", "max_years", "initial_strength"],
    "properties": {
        "decay_model": {"type": "string", "enum": ["exponential", "sigmoid"]},
        "max_years": {"type": "integer", "minimum": 1},
        "initial_strength": {"type": "number", "minimum": 0, "maximum": 1},
        "daily_apply": {"type": "boolean"},
        "manual": {
            "type": "object",
            "properties": {
                "reinforce_step": {"type": "number"},
                "advance_step": {"type": "number"}
            }
        }
    },
    "additionalProperties": True
}

def load_and_validate(path: Path, schema: Dict[str, Any]) -> Tuple[Dict[str, Any], list]:
    data = json.loads(path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda e: e.path)
    return data, errors
