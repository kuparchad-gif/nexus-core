```python
import streamlit as st
import json
import os
import re
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
import base64
from ..utils.logging_config import logger

def validate_api_key(key: str, provider: str) -> bool:
    if not key:
        return True  # Allow empty for optional keys
    patterns = {
        "alpaca": r"^[A-Z0-9]{20}$",
        "polygon": r"^[a-zA-Z0-9_-]{24}$",
        "moralis": r"^[a-zA-Z0-9]{32}$",
        "infura": r"^[0-9a-f]{32}$",
        "alchemy": r"^[a-zA-Z0-9_-]{32}$",
        "alpha_vantage": r"^[A-Z0-9]{16}$",
        "fmp": r"^[a-z0-9]{32}$"
    }
    pattern = patterns.get(provider.lower())
    if pattern and not re.match(pattern, key):
        return False
    return True

def save_api_config(config: dict):
    try:
        # Validate keys
        for provider, value in config.items():
            if isinstance(value, dict):
                for subkey, val in value.items():
                    if not validate_api_key(val, provider):
                        raise ValueError(f"Invalid {provider} {subkey} format")
            else:
                if not validate_api_key(value, provider):
                    raise ValueError(f"Invalid {provider} key format")
        
        # Encrypt and store
        salt = base64.urlsafe_b64encode(os.urandom(16))
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=390000,
            backend=default_backend()
        )
        key = base64.urlsafe_b64encode(kdf.derive(os.getenv("MASTER_PASSWORD", "changeme").encode()))
        fernet = Fernet(key)
        encrypted = fernet.encrypt(json.dumps(config).encode())
        Path(".api_vault").write_bytes(salt + b"::" + encrypted)
        logger.info("API keys encrypted and stored in .api_vault")
        st.success("API keys encrypted and stored securely")
    except Exception as e:
        logger.error(f"Failed to save API config: {e}")
        st.error(f"Failed to save API keys: {e}")

def api_keys_tab():
    st.header("🔑 API Key Vault")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Trading APIs")
        alpaca_key = st.text_input("Alpaca API Key", type="password")
        alpaca_secret = st.text_input("Alpaca Secret", type="password")
        ibkr_port = st.number_input("IBKR Port", 4001, 4003, 4001)
        
    with col2:
        st.subheader("Data Providers")
        polygon_key = st.text_input("Polygon API Key", type="password")
        alpha_vantage_key = st.text_input("Alpha Vantage Key", type="password")
        fmp_key = st.text_input("FMP Key", type="password")
    
    st.subheader("Crypto & DeFi")
    moralis_key = st.text_input("Moralis API Key", type="password")
    infura_key = st.text_input("Infura Project ID", type="password")
    alchemy_key = st.text_input("Alchemy API Key", type="password")
    
    if st.button("💾 Save API Configuration", type="primary"):
        save_api_config({
            "alpaca": {"key": alpaca_key, "secret": alpaca_secret},
            "polygon": polygon_key,
            "moralis": moralis_key,
            "infura": infura_key,
            "alchemy": alchemy_key,
            "alpha_vantage": alpha_vantage_key,
            "fmp": fmp_key
        })
```