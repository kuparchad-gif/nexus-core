from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from loguru import logger


@dataclass
class ModelInfo:
    base_model: str

    @classmethod
    def from_dir(cls, model_dir: Path):
        with open(model_dir / "metadata.json", "r", encoding="utf-8") as file:
            data = json.load(file)
        return ModelInfo(base_model=data["bert_type"])


class ONNXModel:
    def __init__(self, model: ort.InferenceSession, model_info: ModelInfo) -> None:
        self.model = model
        self.model_info = model_info
        self.model_path = Path(model._model_path)  # type: ignore
        self.model_name = self.model_path.name

        self.providers = model.get_providers()

        if self.providers[0] in ["CUDAExecutionProvider", "TensorrtExecutionProvider"]:
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.io_types = {
            "input_ids": np.int32,
            "attention_mask": np.bool_
        }

        self.input_names = [el.name for el in model.get_inputs()]
        self.output_name = model.get_outputs()[0].name

    @staticmethod
    def load_session(
        path: str | Path,
        provider: str = "CPUExecutionProvider",
        session_options: ort.SessionOptions | None = None,
        provider_options: dict[str, Any] | None = None,
    ) -> ort.InferenceSession:
        providers = [provider]
        if provider == "TensorrtExecutionProvider":
            providers.append("CUDAExecutionProvider")
        elif provider == "CUDAExecutionProvider":
            providers.append("CPUExecutionProvider")

        if not isinstance(path, str):
            path = Path(path) / "model.onnx"

        providers_options = None
        if provider_options is not None:
            providers_options = [provider_options] + [{} for _ in range(len(providers) - 1)]

        session = ort.InferenceSession(
            str(path),
            providers=providers,
            sess_options=session_options,
            provider_options=providers_options,
        )
        logger.info("Session loaded")
        return session

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> ONNXModel:
        return ONNXModel(ONNXModel.load_session(model_dir), ModelInfo.from_dir(model_dir))

    def __call__(self, **model_inputs: np.ndarray):
        model_inputs = {
            input_name: tensor.astype(self.io_types[input_name]) for input_name, tensor in model_inputs.items()
        }

        return self.model.run([self.output_name], model_inputs)[0]
