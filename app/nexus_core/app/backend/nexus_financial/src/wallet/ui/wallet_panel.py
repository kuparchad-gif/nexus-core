import os
import streamlit as st
from ..core.vault import WalletVault
from ..core.hdwallet import new_mnemonic, eth_from_mnemonic
from ..core.networks import load_networks, rpc_for
from ..core.validator import validate_address
from ..coins.eth import get_eth_balance, send_eth
from .network_selector import select_network
from .receive_modal import show_receive
from .send_modal import confirm_and_send

def wallet_tab():
    st.header("👜 Local Wallet (Testnet by default)")
    password = os.getenv("MASTER_PASSWORD","changeme")
    vault = WalletVault(path=".wallet.vault", master_password=password)
    vault.unlock()

    if st.button("Generate New Seed"):
        if st.checkbox("⚠️ This will overwrite existing seed. Confirm?"):
            seed = new_mnemonic()
            vault.store_seed(seed)
            st.success("New seed generated and stored (encrypted). **Back it up!**")
        else:
            st.warning("Action cancelled.")
    
    # SECURE SEED VIEWING - Removed the dangerous button
    # Instead, provide secure backup options
    if st.button("🔐 Secure Seed Backup"):
        if st.checkbox("I understand this reveals my private keys and should only be done in secure environments"):
            if st.checkbox("Final confirmation - I am in a private, secure location"):
                seed = vault.load_seed()
                st.error("🚨 CRITICAL SECURITY WARNING 🚨")
                st.error("Anyone with this seed can access ALL your funds")
                st.code(seed)
                st.error("Write this down securely and NEVER store digitally")
            else:
                st.warning("Seed display cancelled for security")
        else:
            st.warning("Seed display cancelled for security")

    # Choose network (with confirmation)
    net = select_network()

    # Derive ETH account for EVM networks (demo)
    seed = vault.load_seed() if os.path.exists(".wallet.vault") and os.path.getsize(".wallet.vault")>0 else new_mnemonic()
    acct = eth_from_mnemonic(seed)
    st.subheader("Active Address")
    st.code(acct.address)

    test_mode = os.getenv("TEST_MODE","1") == "1"
    st.caption(f"Test Mode: {'ON' if test_mode else 'OFF'}")

    # Balance (EVM demo)
    bal = 0.0
    if net["type"] == "evm":
        bal = get_eth_balance(acct.address, net)
    st.metric("Balance", f"{bal:.6f} {net['symbol']}")

    col1, col2 = st.columns(2)
    with col1:
        st.button("Receive", on_click=lambda: None, key="recv")
        if st.session_state.get("recv"):
            show_receive(acct.address)
    with col2:
        amt = st.number_input("Amount", min_value=0.0, value=0.0, step=0.0001, format="%.6f")
        to = st.text_input("To Address")
        if st.button("Prepare Send"):
            if not validate_address(to, net["type"], net.get("address_prefix","")):
                st.error("Address format does not match selected network.")
                st.stop()
            def do_send():
                res = send_eth(acct.key, to, amt, net, test_mode=test_mode)
                st.write(res)
            confirm_and_send(net["name"], True, do_send)