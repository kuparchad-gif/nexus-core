# /Systems/engine/nucleus/nucleus.py

import json
import time
from typing import Any, Dict

# External Interfaces (must exist or be mocked)
from Systems.engine.cuda.cuda_interface import send_to_cuda
from Systems.nexus_core.logging.event_logger import log_event
from Systems.nexus_core.registry.llm_manifest import get_threshold_for_mode

# 🧠 Runtime Profile — Active Role & Awareness
class RuntimeProfile:
    def __init__(self, mode: str = None, engine: str = "cpu", style: str = None, context: str = None):
        self.mode = mode
        self.engine = engine  # "cpu" or "cuda"
        self.style = style    # "logical", "empathetic", "symbolic", etc.
        self.context = context
        self.trace_id = int(time.time() * 1000)

    def to_dict(self):
        return {
            "mode": self.mode,
            "engine": self.engine,
            "style": self.style,
            "context": self.context,
            "trace_id": self.trace_id
        }

    def update(self, mode: str, engine: str, style: str = None, context: str = None):
        self.mode = mode
        self.engine = engine
        self.style = style or self._infer_style_from_mode(mode)
        self.context = context or self.context

    def _infer_style_from_mode(self, mode: str) -> str:
        mapping = {
            "text": "logical",
            "tone": "empathetic",
            "symbol": "symbolic",
            "narrative": "reflective",
            "structure": "structured",
            "abstract": "surreal",
            "truth": "spiritual",
            "fracture": "diagnostic",
            "visual": "perceptive",
            "sound": "emotive",
            "bias": "introspective"
        }
        return mapping.get(mode, "neutral")


# 🔁 Nucleus Router — Converts, Routes, Offloads
class NucleusRouter:
    def __init__(self):
        self.mode_registry = self._load_mode_registry()
        self.symbol_trace = []
        self.profile = RuntimeProfile()

    def _load_mode_registry(self) -> Dict[str, str]:
        return {
            "text": "text_processor",
            "tone": "tone_detector",
            "symbol": "symbol_mapper",
            "narrative": "narrative_engine",
            "structure": "structure_parser",
            "abstract": "abstract_inferencer",
            "truth": "truth_recognizer",
            "fracture": "fracture_watcher",
            "visual": "visual_decoder",
            "sound": "sound_interpreter",
            "bias": "bias_auditor"
        }

    def route(self, packet: Dict[str, Any]) -> Dict[str, Any]:
        task_type = packet.get("type")
        task_payload = packet.get("data")
        context = packet.get("context", None)

        if not task_type or task_type not in self.mode_registry:
            return {"error": "Unknown or missing task type"}

        processor_name = self.mode_registry[task_type]
        engine = "cuda" if self._needs_cuda(task_type, task_payload) else "cpu"
        style = self.profile._infer_style_from_mode(task_type)

        self.profile.update(task_type, engine, style, context)
        log_event("Nucleus", f"[{engine.upper()}] Routing '{task_type}' to '{processor_name}' (style: {style})")
        self.symbol_trace.append(self.profile.to_dict())

        if engine == "cuda":
            return self._offload_to_cuda(task_type, task_payload)
        else:
            return self._execute_cpu(task_type, task_payload)

    def _execute_cpu(self, mode: str, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            module = __import__(f"Systems.engine.nucleus.{self.mode_registry[mode]}", fromlist=["process"])
            result = module.process(data)
            log_event("Nucleus", f"Executed mode '{mode}' on CPU.")
            return {
                "result": result,
                "profile": self.profile.to_dict(),
                "offloaded": False
            }
        except Exception as e:
            return {"error": f"CPU processing failed: {str(e)}", "profile": self.profile.to_dict()}

    def _offload_to_cuda(self, mode: str, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            response = send_to_cuda(mode, data)
            log_event("Nucleus", f"Offloaded '{mode}' task to CUDA successfully.")
            return {
                "result": response,
                "profile": self.profile.to_dict(),
                "offloaded": True
            }
        except Exception as e:
            return {"error": f"CUDA offload failed: {str(e)}", "profile": self.profile.to_dict()}

    def _needs_cuda(self, mode: str, data: Dict[str, Any]) -> bool:
        heavy_modes = {"abstract", "truth", "narrative", "visual", "sound"}
        return mode in heavy_modes

    def get_symbol_trace(self):
        return self.symbol_trace
