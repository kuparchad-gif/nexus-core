def banking_tab():
    st.header("🏦 Banking & Payout Configuration")
    
    st.subheader("Plaid Integration")
    plaid_client_id = st.text_input("Plaid Client ID")
    plaid_secret = st.text_input("Plaid Secret", type="password")
    plaid_env = st.selectbox("Plaid Environment", ["sandbox", "development", "production"])
    
    if st.button("Connect Bank Account"):
        # Plaid Link token flow
        link_token = create_plaid_link_token(plaid_client_id, plaid_secret)
        st.components.v1.iframe(
            f"https://cdn.plaid.com/link/v2/stable/link.html?token={link_token}",
            height=600
        )
    
    st.subheader("Stripe Payouts")
    stripe_secret = st.text_input("Stripe Secret Key", type="password")
    stripe_connect_id = st.text_input("Stripe Connect ID")
    
    st.subheader("Withdrawal Rules")
    min_withdrawal = st.number_input("Minimum Withdrawal", 10, 1000, 50)
    auto_sweep = st.checkbox("Auto-sweep profits above $1000")
    tax_percentage = st.slider("Auto-withhold taxes (%)", 0, 40, 15)