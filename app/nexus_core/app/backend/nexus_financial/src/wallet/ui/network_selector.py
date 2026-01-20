import streamlit as st
from ..core.networks import load_networks
from ..core.validator import validate_address

def select_network():
    nets = load_networks()
    labels = [f"{n['name']} ({n['symbol']})" for n in nets]
    idx = st.selectbox("Network", list(range(len(labels))), format_func=lambda i: labels[i])
    chosen = nets[idx]
    key = "confirmed_network"
    if key not in st.session_state or st.session_state[key] != chosen["key"]:
        st.warning(f"You selected **{chosen['name']}**. Please confirm to proceed.")
        if st.button(f"✅ Yes, use {chosen['name']}"):
            st.session_state[key] = chosen["key"]
            st.success(f"Locked to {chosen['name']}")
        else:
            st.stop()
    return chosen
