# src/service/cognikubes/berts/bert_layer_fixed.py
"""
Compatibility adapter for legacy imports in Acidemikubes:
`from bert_layer_fixed import BertLayerStub`

We map the stub's two used methods to QwenHelperService equivalents.
"""

# Keeps your existing import stable:
# from bert_layer_fixed import BertLayerStub
from .qwen_helper_service import load_qwen_helper

_qwen = None

class BertLayerStub:
    def __init__(self, config_path: str | None = None):
        global _qwen
        if _qwen is None:
            _qwen = load_qwen_helper(config_path)
        self._inner = _qwen

    def process_input(self, text: str):
        return self._inner.process_input(text)

    def classify(self, text: str, labels):
        return self._inner.classify(text, labels)