# examples/python/tts_polly_ssml.py
# AWS Polly SSML example (placeholders — set AWS creds in env or config)
import boto3, uuid, sys
TEXT = "Log streams aligned. Beginning analysis." if len(sys.argv)<2 else sys.argv[1]
VOICE_ID = "Brian"  # replace with your preferred British/Nordic voice
polly = boto3.client("polly")
resp = polly.synthesize_speech(
    Text=f"<speak><prosody rate='-10%' pitch='-2st'>{TEXT}</prosody></speak>",
    TextType="ssml",
    VoiceId=VOICE_ID,
    Engine="neural",
    OutputFormat="mp3"
)
out = f"polly_tts_{uuid.uuid4().hex}.mp3"
with open(out, "wb") as f: f.write(resp["AudioStream"].read())
print("Saved:", out)
