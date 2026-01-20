def agents_tab():
    st.header("🤖 Trading Agents Configuration")
    
    st.subheader("GPT Agent Settings")
    openai_key = st.text_input("OpenAI API Key", type="password")
    agent_personality = st.selectbox("Agent Personality", [
        "Conservative Investor", "Aggressive Trader", "Quant Analyst", 
        "Risk Manager", "Market Maker", "Custom"
    ])
    
    st.subheader("Agent Permissions")
    col1, col2 = st.columns(2)
    with col1:
        can_analyze = st.checkbox("Allow market analysis", True)
        can_suggest = st.checkbox("Allow trade suggestions", True)
        can_monitor = st.checkbox("24/7 market monitoring", True)
    with col2:
        can_execute = st.checkbox("Allow paper trading", True)
        can_withdraw = st.checkbox("Allow payout suggestions", False)
        risk_checks = st.checkbox("Enable risk validation", True)
    
    st.subheader("Auto-Trading Rules")
    max_position_size = st.number_input("Max Position Size (%)", 1, 100, 10)
    daily_loss_limit = st.number_input("Daily Loss Limit ($)", 100, 10000, 1000)
    trading_hours = st.multiselect("Active Trading Hours", 
        ["Pre-market", "Regular", "After-hours", "24/7"])