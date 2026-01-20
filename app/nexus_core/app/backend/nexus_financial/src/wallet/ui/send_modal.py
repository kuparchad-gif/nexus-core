import streamlit as st
from ..core.validator import validate_address

def confirm_and_send(network_name: str, address_ok: bool, on_send):
    st.warning(f"⚠️ You are about to send on **{network_name}**. Using the wrong network can result in loss of funds.")
    ok = st.checkbox("I understand and confirm")
    if not ok:
        st.stop()
    if st.button("Send Now"):
        return on_send()
