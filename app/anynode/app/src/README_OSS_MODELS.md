# OSS Models Bootstrap (Hugging Face + Firmware-ready)

This pack restores the plan to keep **sovereign OSS models** alongside your Nexus microcells.
It gives you a registry, download scripts, and a tiny “firmware packer” for on-device micro-models.

## 0) Prereqs
- Install Python 3.10+
- `pip install huggingface_hub[cli] hf-transfer onnx onnxruntime`
- Create a **Hugging Face token**: https://huggingface.co/settings/tokens  (read access)
- (Windows) PowerShell: `setx HF_HOME C:\hf-cache`  (or choose a drive with space)
- (Speed) `setx HF_HUB_ENABLE_HF_TRANSFER 1`

## 1) Configure registry
Edit `models/registry.yaml` to enable/disable models per capability or swap IDs.

## 2) Login + Download
```powershell
# one-time login (stores token in keyring or env if set)
python scripts/hf_login.py --token YOUR_HF_TOKEN

# download everything marked "enabled: true" into ./models/_hub/
python scripts/download_models.py

# or just one capability (e.g., embeddings)
python scripts/download_models.py --cap embeddings

# verify sizes & SHAs
python scripts/verify_registry.py
```

## 3) Use in microcells
Set envs in compose/k8s:
```
MODELS_ROOT=/models/_hub
EMBED_MODEL=BAAI/bge-m3
LLM_PRIMARY=meta-llama/Llama-3.1-8B-Instruct
LLM_SECONDARY=mistralai/Mistral-7B-Instruct-v0.3
VISION_MODEL=Qwen/Qwen2-VL-7B-Instruct
ASR_MODEL=Systran/faster-whisper-large-v3
```
Your gateway/anynode/viren/loki can read `models/registry.yaml` directly (see `src/aethereal/gateway/model_registry.py`).

## 4) Firmware pack (tiny models)
```powershell
# package a small model (e.g., MiniLM) into a deployable "firmware" bundle (.zip)
python scripts/pack_firmware.py --model sentence-transformers/all-MiniLM-L6-v2 --out dist/firmware_minilm.zip
```
The bundle contains a manifest, ONNX weights, and a small runner stub.

## Notes
- **Licenses** vary by model. The script prints the license string found in the Hugging Face metadata; review before commercial use.
- Some models (e.g., Llama 3.1) require accepting a specific EULA on the model page. Run `hf_hub_download` after you’ve been granted access.
