```python
import streamlit as st
import json
from pathlib import Path
from datetime import datetime
from ..utils.logging_config import logger

def generate_security_report():
    try:
        report = {
            "2fa_enabled": st.session_state.get("require_2fa", False),
            "ip_whitelist": st.session_state.get("ip_whitelist", ""),
            "rate_limits": st.session_state.get("rate_limits", 100),
            "last_audit": datetime.now().isoformat(),
            "auto_lock_enabled": st.session_state.get("auto_lock", True)
        }
        Path("artifacts/security_report.json").write_text(json.dumps(report, indent=2))
        logger.info("Security report exported to artifacts/security_report.json")
        st.success("Security report exported to artifacts/security_report.json")
    except Exception as e:
        logger.error(f"Failed to generate security report: {e}")
        st.error(f"Failed to generate report: {e}")

def security_tab():
    st.header("🔒 Security & Access Control")
    
    st.subheader("Authentication")
    require_2fa = st.checkbox("Require 2FA for trades", True, key="require_2fa")
    session_timeout = st.slider("Session Timeout (minutes)", 15, 480, 60)
    
    st.subheader("API Security")
    ip_whitelist = st.text_area("IP Whitelist (one per line)", key="ip_whitelist")
    rate_limits = st.number_input("API Calls per Minute", 10, 1000, 100, key="rate_limits")
    
    st.subheader("Wallet Security")
    hw_wallet = st.selectbox("Hardware Wallet", 
        ["None", "Ledger", "Trezor", "Coldcard"])
    auto_lock = st.checkbox("Auto-lock wallet after 5min", True, key="auto_lock")
    
    if st.button("🛡️ Export Security Audit", type="secondary"):
        generate_security_report()
```