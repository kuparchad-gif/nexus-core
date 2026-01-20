import urequests, ujson
ROUTER="{{ROUTER_URL}}"

def send(text: str, mode=None, max_new_tokens=128):
    payload={"text":text, "mode":mode, "max_new_tokens":max_new_tokens}
    r=urequests.post(ROUTER, data=ujson.dumps(payload),
                     headers={"Content-Type":"application/json"})
    try:
        return r.json()
    finally:
        r.close()
