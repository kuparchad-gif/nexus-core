# Nexus Voices & Protocols v1

This kit gives you:
- **Voice personas** for Loki (Viking) and Viren (Hopkins‑inspired sage)
- **Protocols** (prime + secondary) for Viren — truth‑first
- **Streaming frames** spec for audio with data controls (SDCF)
- **TTS examples** for Azure, AWS Polly, and ElevenLabs (placeholders — add keys)

## Quick start
1) Pick a provider (Azure, Polly, ElevenLabs, or Coqui). Add your API keys as env vars.
2) Open `voices/voices.yaml` and select provider + voice IDs (placeholders provided).
3) Wire your agent runtime to:
   - render **SSML** from persona settings; and/or
   - stream via provider’s WebSocket/HTTP using our **SDCF** frames.
4) Drop the `system_prompt.md` files into each CogniKube as the agent's system message.
