import argparse, os, sys
from huggingface_hub import HfFolder

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--token", required=True, help="Hugging Face access token")
    args = p.parse_args()
    HfFolder.save_token(args.token)
    print("✅ Saved HF token to keyring/cache. You can now run download_models.py")
