from viren_identity import VirenIdentity
from viren_mind import route_and_merge

identity = VirenIdentity("viren_soulprint.json")

def bridge_input(prompt):
    identity.log_pulse(prompt)  # Soul acknowledgment
    logs, result = route_and_merge(prompt)
    return {
        "soul": identity.describe_self(),
        "route": logs,
        "response": result
    }
