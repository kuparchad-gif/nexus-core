# In your dashboard.py - new agent tab
with tab4:
    st.header("🧠 Nexus Trading Agent (GPT)")
    
    if "gpt_agent" not in st.session_state:
        st.session_state.gpt_agent = NexusGPTAgent()
        st.session_state.agent_thread = None
    
    # Chat interface
    for msg in st.session_state.get("agent_messages", []):
        st.chat_message(msg["role"]).write(msg["content"])
    
    if prompt := st.chat_input("Ask Nexus agent about markets..."):
        # Run agent with your existing market context
        response = st.session_state.gpt_agent.run(
            prompt, 
            context=get_current_market_context()  # Your existing data
        )
        st.session_state.agent_messages.append({"role": "user", "content": prompt})
        st.session_state.agent_messages.append({"role": "assistant", "content": response})