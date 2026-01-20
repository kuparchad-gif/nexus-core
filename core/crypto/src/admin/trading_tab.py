def trading_tab():
    st.header("📊 Advanced Trading Configuration")
    
    st.subheader("Strategy Parameters")
    momentum_period = st.slider("Momentum Period (days)", 5, 50, 20)
    risk_adjustment = st.select_slider("Risk Appetite", 
        ["Very Low", "Low", "Medium", "High", "Very High"])
    
    st.subheader("Order Execution")
    execution_venue = st.radio("Primary Execution", 
        ["Alpaca", "Interactive Brokers", "Coinbase", "Binance"])
    slippage_tolerance = st.slider("Slippage Tolerance (%)", 0.1, 5.0, 1.0)
    
    st.subheader("Rebalancing Rules")
    rebalance_frequency = st.selectbox("Rebalance Frequency", 
        ["Daily", "Weekly", "Bi-weekly", "Monthly", "On Signal"])
    threshold_rebalance = st.checkbox("Threshold-based rebalancing", True)
    if threshold_rebalance:
        threshold = st.slider("Rebalance Threshold (%)", 1, 20, 5)