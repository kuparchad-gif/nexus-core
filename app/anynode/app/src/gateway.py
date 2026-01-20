# NIM TCP Gateway
import os, json, struct, asyncio
from .codec import encode, decode
from .adapters.openai_compat import OpenAICompat
DEFAULT_HOST = os.environ.get("NIM_HOST","0.0.0.0")
DEFAULT_PORT = int(os.environ.get("NIM_PORT","8787"))
DATA_TILES   = int(os.environ.get("NIM_TILES","48"))
OAI_URL      = os.environ.get("NIM_OAI_URL","http://localhost:1234")
OAI_KEY      = os.environ.get("NIM_OAI_KEY","")
OAI_MODEL    = os.environ.get("NIM_OAI_MODEL","gpt-4o-mini")
ADAPTER = OpenAICompat(OAI_URL, OAI_KEY, OAI_MODEL)
async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    try:
        while True:
            hdr = await reader.readexactly(4)
            (n,) = struct.unpack(">I", hdr)
            data = await reader.readexactly(n)
            try:
                payload = decode(data, data_tiles_per_frame=DATA_TILES)
                req = json.loads(payload.decode("utf-8", errors="ignore"))
            except Exception as e:
                err = {"error": f"decode/json failed: {e}"}
                out = encode(json.dumps(err).encode("utf-8"), data_tiles_per_frame=DATA_TILES)
                writer.write(struct.pack(">I", len(out)) + out)
                await writer.drain()
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
            out = encode(json.dumps(resp).encode("utf-8"), data_tiles_per_frame=DATA_TILES)
            writer.write(struct.pack(">I", len(out)) + out)
            await writer.drain()
    except asyncio.IncompleteReadError:
        pass
    finally:
        writer.close()
        await writer.wait_closed()
async def serve(host=DEFAULT_HOST, port=DEFAULT_PORT):
    server = await asyncio.start_server(_handle, host, port)
    addrs = ", ".join(str(s.getsockname()) for s in server.sockets)
    print(f"NIM gateway listening on {addrs}  (tiles/frame={DATA_TILES})")
    async with server:
        await server.serve_forever()
if __name__ == "__main__":
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        pass
