# TLS/mTLS + AEAD NIM gateway
import os, json, struct, asyncio, ssl
from nim.codec import encode, decode
from nim.secure import new_session, open as aead_open, seal as aead_seal, Session
from nim.adapters.openai_compat import OpenAICompat

DEFAULT_HOST = os.environ.get("NIM_HOST","0.0.0.0")
DEFAULT_PORT = int(os.environ.get("NIM_PORT","8787"))
DATA_TILES   = int(os.environ.get("NIM_TILES","48"))
OAI_URL      = os.environ.get("NIM_OAI_URL","http://localhost:1234")
OAI_KEY      = os.environ.get("NIM_OAI_KEY","")
OAI_MODEL    = os.environ.get("NIM_OAI_MODEL","gpt-4o-mini")

TLS_CERT     = os.environ.get("NIM_TLS_CERT","/etc/ssl/nim-gw.crt")
TLS_KEY      = os.environ.get("NIM_TLS_KEY","/etc/ssl/nim-gw.key")
TLS_CLIENT_CA= os.environ.get("NIM_TLS_CLIENT_CA")
MASTER_KEY_HEX = os.environ.get("NIM_AEAD_MASTER")
CONTEXT      = os.environ.get("NIM_CONTEXT","NEXUS-NIM")

ADAPTER = OpenAICompat(OAI_URL, OAI_KEY, OAI_MODEL)

def _ctx():
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_3
    ctx.load_cert_chain(certfile=TLS_CERT, keyfile=TLS_KEY)
    if TLS_CLIENT_CA:
        ctx.load_verify_locations(cafile=TLS_CLIENT_CA)
        ctx.verify_mode = ssl.CERT_REQUIRED
    return ctx

async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    if not MASTER_KEY_HEX: 
        raise RuntimeError("NIM_AEAD_MASTER not set (32B hex)")
    master = bytes.fromhex(MASTER_KEY_HEX)
    server_sess = new_session(master, CONTEXT.encode(), b"S")
    client_sess = Session(server_sess.key, server_sess.sess_id, 0, b"C")

    try:
        while True:
            hdr = await reader.readexactly(4)
            (n,) = struct.unpack(">I", hdr)
            wire = await reader.readexactly(n)
            sealed = decode(wire, data_tiles_per_frame=DATA_TILES)
            try:
                payload = aead_open(client_sess, sealed)
                req = json.loads(payload.decode("utf-8", errors="ignore"))
            except Exception as e:
                err = {"error": f"decrypt/json failed: {e}"}
                out = aead_seal(server_sess, json.dumps(err).encode("utf-8"))
                wire = encode(out, data_tiles_per_frame=DATA_TILES)
                writer.write(struct.pack(">I", len(wire)) + wire); await writer.drain()
                continue

            prompt = req.get("prompt","")
            temp = float(req.get("temperature", 0.2))
            system = req.get("system", "You are a helpful assistant.")
            loop = asyncio.get_running_loop()
            try:
                text = await loop.run_in_executor(None, lambda: ADAPTER.chat(prompt, temp, system))
                resp = {"text": text}
            except Exception as e:
                resp = {"error": str(e)}

            sealed_out = aead_seal(server_sess, json.dumps(resp).encode("utf-8"))
            wire_out = encode(sealed_out, data_tiles_per_frame=DATA_TILES)
            writer.write(struct.pack(">I", len(wire_out)) + wire_out); await writer.drain()
    except asyncio.IncompleteReadError:
        pass
    finally:
        writer.close(); await writer.wait_closed()

async def serve(host=DEFAULT_HOST, port=DEFAULT_PORT):
    server = await asyncio.start_server(_handle, host, port, ssl=_ctx())
    addrs = ", ".join(str(s.getsockname()) for s in server.sockets)
    print(f"NIM gateway (TLS1.3 + AEAD) on {addrs}  tiles/frame={DATA_TILES}")
    async with server: await server.serve_forever()

if __name__ == "__main__":
    try: import asyncio; asyncio.run(serve())
    except KeyboardInterrupt: pass
