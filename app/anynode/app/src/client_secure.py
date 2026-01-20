# Secure client: TLS + NIM + AEAD
import os, json, socket, struct, ssl
from nim.codec import encode, decode
from nim.secure import new_session, seal as aead_seal, open as aead_open, Session

HOST = os.environ.get("NIM_HOST","127.0.0.1")
PORT = int(os.environ.get("NIM_PORT","8787"))
TILES = int(os.environ.get("NIM_TILES","48"))
MASTER_KEY_HEX = os.environ.get("NIM_AEAD_MASTER")
CONTEXT = os.environ.get("NIM_CONTEXT","NEXUS-NIM")

TLS_CA   = os.environ.get("NIM_TLS_CA")
TLS_CERT = os.environ.get("NIM_TLS_CLIENT_CERT")
TLS_KEY  = os.environ.get("NIM_TLS_CLIENT_KEY")

def tls_ctx():
    ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=TLS_CA if TLS_CA else None)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_3
    if TLS_CERT and TLS_KEY:
        ctx.load_cert_chain(TLS_CERT, TLS_KEY)
    return ctx

def ask(prompt: str):
    if not MASTER_KEY_HEX: raise RuntimeError("NIM_AEAD_MASTER not set")
    master = bytes.fromhex(MASTER_KEY_HEX)
    client_sess = new_session(master, CONTEXT.encode(), b"C")
    server_sess = Session(client_sess.key, client_sess.sess_id, 0, b"S")

    req = json.dumps({"prompt": prompt}).encode("utf-8")
    sealed = aead_seal(client_sess, req)
    wire = encode(sealed, data_tiles_per_frame=TILES)

    with socket.create_connection((HOST, PORT), timeout=10) as raw:
        with tls_ctx().wrap_socket(raw, server_hostname=HOST) as s:
            s.sendall(struct.pack(">I", len(wire)) + wire)
            hdr = s.recv(4); 
            if len(hdr) < 4: raise RuntimeError("short read")
            (n,) = struct.unpack(">I", hdr)
            data = b""
            while len(data) < n:
                chunk = s.recv(n-len(data))
                if not chunk: break
                data += chunk

    sealed_resp = decode(data, data_tiles_per_frame=TILES)
    plain = aead_open(server_sess, sealed_resp)
    return json.loads(plain.decode("utf-8", errors="ignore"))

if __name__ == "__main__":
    import sys
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Test encrypted Nexus path."
    print(ask(prompt))
