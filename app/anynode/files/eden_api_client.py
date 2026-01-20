from Systems.engine.comms.eden_api_guard import EdenAPIGuard

api_guard  =  EdenAPIGuard()

def send_request(model, payload):
    tokens_estimate  =  estimate_tokens(payload)

    allowed, message  =  api_guard.allow_request(tokens_estimate)
    if not allowed:
        print(f"🚫 Blocked API request: {message}")
        return None

    # Otherwise continue sending request normally
    response  =  actually_send_request(model, payload)
    return response
