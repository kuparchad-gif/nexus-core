import os, ssl, nats

def tls_ctx():
    ctx = ssl.create_default_context(cafile="/certs/ca/ca.crt")
    ctx.load_cert_chain("/certs/client/client.crt","/certs/client/client.key")
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    return ctx

async def connect():
    url = os.getenv("MCP_NATS_URL","nats://host.containers.internal:4222")
    return await nats.connect(servers=[url], tls=tls_ctx())

