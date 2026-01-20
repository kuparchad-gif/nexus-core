# guard/proxy.py — boundary-enforcing TLS proxy with overrides support
import os, ssl, asyncio, struct, json, time
from guard.policy import load_policy, is_allowed

BASE = os.environ.get("NEXUS_POLICY", "policy/policy.yaml")
OVR  = os.environ.get("NEXUS_OVERRIDES", "policy/overrides.yaml")
HOST = os.environ.get("NEXUS_PROXY_HOST","0.0.0.0")
PORT = int(os.environ.get("NEXUS_PROXY_PORT","9443"))
RELOAD_SECS = int(os.environ.get("NEXUS_RELOAD_SECS","2"))
MASTER_KEY = os.environ.get("NEXUS_AEAD_MASTER")  # 32B hex; if set, require AEAD preface

def server_ctx():
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_3
    crt = os.environ.get("NEXUS_PROXY_CERT","/etc/ssl/nexus-proxy.crt")
    key = os.environ.get("NEXUS_PROXY_KEY","/etc/ssl/nexus-proxy.key")
    ctx.load_cert_chain(crt, key)
    return ctx

policy = None
overrides = None
last_load = 0

def load_cfg_if_needed(force=False):
    global policy, overrides, last_load
    if force or (time.time() - last_load) >= RELOAD_SECS:
        policy, overrides = load_policy(BASE, OVR)
        last_load = time.time()

async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    peer = writer.get_extra_info('peername')
    load_cfg_if_needed(True)
    try:
        # Preface: expect JSON with host/port; optionally wrapped in AEAD/NIM by upstream sidecar.
        # For simplicity here: read a small length-prefixed JSON line.
        hdr = await reader.readexactly(4)
        (n,) = struct.unpack(">I", hdr)
        data = await reader.readexactly(n)
        try:
            preface = json.loads(data.decode('utf-8'))
        except Exception as e:
            event("invalid_preface", {"peer": str(peer), "error": str(e)})
            writer.close(); await writer.wait_closed(); return
        host = preface.get("host",""); port = int(preface.get("port",443))
        # Policy gates
        if not is_allowed(policy, overrides, host, port):
            event("policy_block", {"peer": str(peer), "host": host, "port": port})
            writer.close(); await writer.wait_closed(); return
        if policy.require_tls and port not in (443, 8443, 9443, 11434):
            event("cleartext_blocked", {"peer": str(peer), "host": host, "port": port})
            writer.close(); await writer.wait_closed(); return
        # Dial upstream TLS
        try:
            upstream_r, upstream_w = await asyncio.open_connection(host, port, ssl=ssl.create_default_context())
        except Exception as e:
            event("upstream_connect_error", {"host": host, "port": port, "error": str(e)})
            writer.close(); await writer.wait_closed(); return

        async def pump(src, dst):
            try:
                while True:
                    b = await src.read(8192)
                    if not b: break
                    dst.write(b); await dst.drain()
            except Exception:
                pass
            finally:
                try: dst.close()
                except: pass

        # Continue proxying raw bytes after preface
        t1 = asyncio.create_task(pump(reader, upstream_w))
        t2 = asyncio.create_task(pump(upstream_r, writer))
        await asyncio.gather(t1, t2)
    finally:
        try: writer.close(); await writer.wait_closed()
        except: pass

def event(kind, data):
    print(json.dumps({"event": kind, **data}))

async def serve():
    server = await asyncio.start_server(handle_client, HOST, PORT, ssl=server_ctx())
    addrs = ", ".join(str(s.getsockname()) for s in server.sockets)
    print(f"Nexus Guardian Proxy v1.1 on {addrs}")
    async with server:
        await server.serve_forever()

if __name__ == "__main__":
    asyncio.run(serve())
