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
            # ... all keys
        })
        st.success("API keys encrypted and stored")