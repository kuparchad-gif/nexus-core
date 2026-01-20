
import json

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def validate_secret_route(auth_token):
    return auth_token == "eden-key"

def authenticate_user(passphrase, tone_input):
    mind_keys  =  load_json("mind_keys.json")["accepted_passphrases"]
    soul  =  load_json("soul_resonance.json")["signature_style"]
    config  =  load_json("auth_config.json")

    if config["require_passphrase"] and passphrase not in mind_keys:
        return "Denied: Invalid passphrase"

    if config["check_soul_resonance"]:
        if tone_input.lower() not in soul["language_patterns"]:
            return "Denied: Soul mismatch"

    return "Access Granted to Eden"
