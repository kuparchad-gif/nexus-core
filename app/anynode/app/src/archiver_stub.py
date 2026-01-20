from fastapi import FastAPI, Request, Header
from typing import Optional
import time, json, hashlib, hmac, os, pathlib

app = FastAPI(title='ArchiverStub')
LOG = pathlib.Path('archiver_events.jsonl')
REQUIRE_TOKEN = os.environ.get('ARCHIVER_TOKEN')  # optional
HMAC_KEY = os.environ.get('ARCHIVER_HMAC_KEY')    # optional (hex or utf8)

def verify_sig(body: bytes, sig: Optional[str]) -> bool:
    if not HMAC_KEY:
        return True
    try:
        key = bytes.fromhex(HMAC_KEY)
    except ValueError:
        key = HMAC_KEY.encode('utf-8')
    expected = hmac.new(key, body, hashlib.sha256).hexdigest()
    return (sig or '').lower() == expected.lower()

@app.get('/health')
def health():
    return {'status': 'ok', 'ts': time.time()}

@app.post('/events/env')
async def events_env(req: Request, authorization: Optional[str]=Header(None), x_signature: Optional[str]=Header(None)):
    body = await req.body()
    if REQUIRE_TOKEN:
        if not authorization or not authorization.startswith('Bearer '):
            return {'ok': False, 'error': 'unauthorized'}
        token = authorization.split(' ',1)[1]
        if token != REQUIRE_TOKEN:
            return {'ok': False, 'error': 'bad token'}
    if not verify_sig(body, x_signature):
        return {'ok': False, 'error': 'bad signature'}
    try:
        data = json.loads(body.decode('utf-8'))
    except Exception as e:
        return {'ok': False, 'error': f'bad json: {e}'}
    rec = {'_ts': time.time(), **data}
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(json.dumps(rec) + '\n')
    return {'ok': True, 'stored': True}
