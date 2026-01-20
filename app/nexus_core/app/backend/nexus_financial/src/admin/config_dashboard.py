# src/admin/config_dashboard.py
import streamlit as st
import yaml
from pathlib import Path

def settings_main():
    st.title("⚙️ Nexus Control Center")
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔑 API Keys", "🎨 Themes", "🏦 Banking", "🤖 Agents", 
        "📊 Trading", "🔒 Security"
    ])
    
    with tab1:
        api_keys_tab()
    with tab2:
        themes_tab() 
    with tab3:
        banking_tab()
    with tab4:
        agents_tab()
    with tab5:
        trading_tab()
    with tab6:
        security_tab()