Catalyst expansion modules added:
- abstract_inferencer.AbstractInferencer  (use when trigger='dream')
- truth_recognizer.TruthRecognizer        (feed with symbol keys via TONGUE)
- fracture_watcher.FractureWatcher        (text contradictions)
- visual_decoder.VisualDecoder            (image token decoding)
- sound_interpreter.SoundInterpreter      (audio features)
- bias_auditor.BiasAuditor                (internal weight skew)

Example wiring in pulse_router.py:
    from .abstract_inferencer import AbstractInferencer
    from .truth_recognizer import TruthRecognizer
    from .fracture_watcher import FractureWatcher
    from .visual_decoder import VisualDecoder
    from .sound_interpreter import SoundInterpreter
    from .bias_auditor import BiasAuditor

    self.mod.update({
        "abstract": AbstractInferencer(),
        "truth": TruthRecognizer(),
        "fracture": FractureWatcher(),
        "visual": VisualDecoder(),
        "sound": SoundInterpreter(),
        "bias": BiasAuditor(),
    })
