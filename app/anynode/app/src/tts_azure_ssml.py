# examples/python/tts_azure_ssml.py
# SSML TTS with Azure (placeholders — set AZURE_TTS_KEY & AZURE_TTS_REGION)
import os, uuid, sys
import requests

AZURE_KEY = os.getenv("AZURE_TTS_KEY","")
AZURE_REGION = os.getenv("AZURE_TTS_REGION","")
VOICE_ID = os.getenv("AZURE_VOICE_ID","en-GB-male-02")  # replace with actual ID in your account
TEXT = "Systems green. Standing by." if len(sys.argv)<2 else sys.argv[1]

ssml = f"""<speak version='1.0' xml:lang='en-GB'>
  <voice name='{VOICE_ID}'>
    <prosody rate='-10%' pitch='-2st'>{TEXT}</prosody>
  </voice>
</speak>"""

if not AZURE_KEY or not AZURE_REGION:
    raise SystemExit("Set AZURE_TTS_KEY and AZURE_TTS_REGION")

url = f"https://{AZURE_REGION}.tts.speech.microsoft.com/cognitiveservices/v1"
headers = {
    "Ocp-Apim-Subscription-Key": AZURE_KEY,
    "Content-Type": "application/ssml+xml",
    "X-Microsoft-OutputFormat": "audio-24khz-48kbitrate-mono-mp3",
    "User-Agent": "nexus-tts-demo"
}
r = requests.post(url, data=ssml.encode("utf-8"), headers=headers, timeout=30)
r.raise_for_status()
out = f"azure_tts_{uuid.uuid4().hex}.mp3"
open(out,"wb").write(r.content)
print("Saved:", out)
