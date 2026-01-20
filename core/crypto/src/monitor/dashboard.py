from src.utils.safe_eval import safe_eval
# UPDATED dashboard.py
import streamlit as st
import pandas as pd
from pathlib import Path
from ..config import ARTIFACTS_DIR, QDRANT_LOCAL_URL, QDRANT_CLOUD_URL, QDRANT_API_KEY
from ..utils.plots import equity_curve_figure, drawdown_figure, weights_heatmap
from ..storage.qdrant_store import _client, COLL, vectorize_row
import numpy as np

st.set_page_config(page_title="Nexus Trading Suite", layout="wide")
st.title("🚀 Nexus Trading & Monitoring Suite")

# NEW: Sidebar navigation
st.sidebar.title("🌌 Nexus Navigator")
app_mode = st.sidebar.selectbox(
    "Select Mode",
    ["📈 Trading Dashboard", "🤖 AI Agent", "⚙️ Control Center"],
    key="app_mode"
)

# NEW: Control Center Import
if app_mode == "⚙️ Control Center":
    from ..admin.config_dashboard import settings_main
    settings_main()
    st.stop()

# NEW: Agent Import  
if app_mode == "🤖 AI Agent":
    from ..agents.gpt_agent import agent_tab
    agent_tab()
    st.stop()

# ORIGINAL DASHBOARD (now with enhanced tabs)
run_dir = ARTIFACTS_DIR / "latest"
metrics_fp = run_dir / "metrics.json"
series_fp = run_dir / "series.parquet"
weights_fp = run_dir / "weights.parquet"

# ENHANCED: Added Agent tab
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Performance", "🔍 Pattern Search", "👜 Wallet", "🧠 Quick Agent"
])

with tab1:
    if not run_dir.exists():
        st.warning("No artifacts found. Run a backtest first.")
        # NEW: Quick setup guide
        if st.button("🚀 Run First Backtest"):
            st.info("""
            **Quick Start:**
            ```bash
            python -m src.app backtest --config config/config.yaml
            ```
            Then refresh this page.
            """)
    else:
        with open(metrics_fp, "r", encoding="utf-8") as f:
            metrics = pd.Series(safe_eval(f.read()))
        series = pd.read_parquet(series_fp)
        weights = pd.read_parquet(weights_fp)

        # ENHANCED: Better metrics layout
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("💰 Ann. Return", f"{metrics['ann_return']:.2%}")
        col2.metric("📊 Ann. Vol", f"{metrics['ann_vol']:.2%}")
        col3.metric("🎯 Sharpe", f"{metrics['sharpe']:.2f}")
        col4.metric("📉 Max DD", f"{metrics['mdd']:.2%}")
        col5.metric("🔄 Win Rate", f"{metrics.get('win_rate', 0):.1%}")

        st.plotly_chart(equity_curve_figure(series['equity']), use_container_width=True)
        st.plotly_chart(drawdown_figure(series['drawdown']), use_container_width=True)
        st.plotly_chart(weights_heatmap(weights), use_container_width=True)

with tab2:
    st.caption("Find historically similar snapshots to a given feature vector (from latest features/signals).")
    
    # ENHANCED: Better vector input
    col1, col2 = st.columns(2)
    with col1:
        sma_diff = st.slider("SMA Difference", -1.0, 1.0, 0.0, 0.01)
        rsi14 = st.slider("RSI (14)", 0.0, 100.0, 50.0, 0.5)
    with col2:
        atr = st.number_input("ATR ($)", value=5.0, help="Average True Range")
        ret1 = st.number_input("1-Day Return %", value=0.0)
    
    vec = [float(sma_diff), float(rsi14), float(atr), float(ret1)]

    # ENHANCED: Search options
    col1, col2, col3 = st.columns(3)
    with col1:
        use_cloud = st.checkbox("Use Qdrant Cloud", value=False)
    with col2:
        k = st.slider("Results Count", 5, 50, 10, 5)
    with col3:
        min_confidence = st.slider("Min Confidence", 0.1, 1.0, 0.7, 0.1)

    client = _client(QDRANT_CLOUD_URL if use_cloud and QDRANT_CLOUD_URL else QDRANT_LOCAL_URL, QDRANT_API_KEY if use_cloud else None)

    if st.button("🔍 Search Similar Patterns", type="primary"):
        try:
            with st.spinner("Searching historical patterns..."):
                res = client.search(collection_name=COLL, query_vector=vec, limit=int(k))
                
                # ENHANCED: Better results display
                matches = []
                for p in res:
                    if p.score >= min_confidence:
                        match_data = {**(p.payload or {}), "confidence": f"{p.score:.1%}"}
                        matches.append(match_data)
                
                if matches:
                    st.subheader(f"🎯 {len(matches)} Pattern Matches Found")
                    df = pd.DataFrame(matches)
                    
                    # NEW: Pattern analysis
                    avg_return = df.get('return', 0.0).mean() if 'return' in df.columns else "N/A"
                    st.metric("📈 Average Historical Return", f"{avg_return:.2%}" if isinstance(avg_return, (int, float)) else avg_return)
                    
                    st.dataframe(df, use_container_width=True)
                else:
                    st.warning("No confident matches found. Try adjusting parameters.")
                    
        except Exception as e:
            st.error(f"Search failed: {e}")

with tab3:
    from ..wallet.ui.wallet_panel import wallet_tab
    wallet_tab()

# NEW: Quick Agent Tab
with tab4:
    st.header("🧠 Nexus Quick Agent")
    st.caption("Get instant AI analysis of your current portfolio and market conditions")
    
    # Simple agent interface without full setup
    agent_query = st.text_area(
        "Ask about your portfolio, trading signals, or market conditions:",
        placeholder="e.g., 'What's the risk in my current portfolio?' or 'Find bullish patterns in QQQ'",
        height=100
    )
    
    col1, col2, col3 = st.columns([1,1,2])
    with col1:
        analyze_current = st.button("📊 Analyze Portfolio")
    with col2:
        scan_markets = st.button("🔍 Scan Markets")
    
    if analyze_current or scan_markets or agent_query:
        try:
            # Simple agent call using existing data
            if run_dir.exists():
                weights = pd.read_parquet(weights_fp)
                current_weights = weights.iloc[-1].to_dict()
                
                if analyze_current:
                    analysis = f"""
                    **Portfolio Analysis:**
                    - **Current Positions:** {len([w for w in current_weights.values() if abs(w) > 0.01])} active
                    - **Largest Position:** {max(current_weights.items(), key=lambda x: abs(x[1]))}
                    - **Concentration Risk:** {'High' if max(current_weights.values()) > 0.3 else 'Medium' if max(current_weights.values()) > 0.2 else 'Low'}
                    - **Suggested Action:** Consider rebalancing if any position exceeds 25% of portfolio.
                    """
                    st.success(analysis)
                
                if scan_markets:
                    st.info("Market scan would analyze current Qdrant patterns and suggest opportunities...")
                    
            else:
                st.warning("Run a backtest first to enable portfolio analysis")
                
        except Exception as e:
            st.error(f"Agent analysis failed: {e}")
    
    # Quick settings
    with st.expander("⚙️ Agent Settings"):
        st.checkbox("Enable real-time monitoring", True)
        st.checkbox("Alert on pattern breaks", True)
        st.slider("Risk tolerance", 1, 5, 3)

# NEW: Status footer
st.sidebar.markdown("---")
st.sidebar.caption(f"🟢 System Online | 📁 {run_dir.name if run_dir.exists() else 'No Data'}")