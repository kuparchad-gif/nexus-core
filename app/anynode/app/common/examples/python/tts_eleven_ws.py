# examples/python/tts_eleven_ws.py
# ElevenLabs streaming WS example (placeholders — set ELEVEN_API_KEY and VOICE_ID)
import os, json, base64, websockets, asyncio, uuid

async def main():
    api_key  =  os.getenv("ELEVEN_API_KEY","")
    voice_id  =  os.getenv("ELEVEN_VOICE_ID","REPLACE_WITH_EL_VOICE_ID")
    if not api_key: raise SystemExit("Set ELEVEN_API_KEY")
    uri  =  f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input?optimize_streaming_latency = 3"
    async with websockets.connect(uri, extra_headers = {"xi-api-key": api_key}) as ws:
        # Begin stream
        await ws.send(json.dumps({"text": "Initializing stream.", "voice_settings":{"stability":0.5,"similarity_boost":0.6}}))
        # Send content frames
        for chunk in ["Systems", " ", "online.", " Standing by."]:
            await ws.send(json.dumps({"text": chunk, "try_trigger_generation": True}))
        await ws.send(json.dumps({"text": "", "try_trigger_generation": True, "flush": True}))
        # Receive audio chunks (base64-encoded)
        idx = 0
        with open(f"eleven_stream_{uuid.uuid4().hex}.mp3","wb") as f:
            async for msg in ws:
                data  =  json.loads(msg)
                if "audio" in data:
                    f.write(base64.b64decode(data["audio"]))
                    idx+ = 1
                if data.get("isFinal"): break
        print("Saved stream output.")
asyncio.run(main())
