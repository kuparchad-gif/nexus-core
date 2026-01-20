# In his existing dashboard.py - add this:
def main():
    st.sidebar.title("Nexus Trader")
    
    menu = st.sidebar.selectbox("Navigation", [
        "📈 Dashboard", "👜 Wallet", "🔍 Similarity", "⚙️ Control Center"
    ])
    
    if menu == "⚙️ Control Center":
        from src.admin.config_dashboard import settings_main
        settings_main()
    else:
        # Existing tabs...