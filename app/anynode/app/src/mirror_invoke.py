
# mirror_invoke.py
from .runtime.council_of_mirrors import CouncilOfMirrors

def invoke_reflection(input_text):
    mirror = CouncilOfMirrors()
    return mirror.summon(input_text)
