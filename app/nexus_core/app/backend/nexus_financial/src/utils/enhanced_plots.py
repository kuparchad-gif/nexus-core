# Enhanced plots.py - Cyberpunk theme
def cyberpunk_equity_curve(equity: pd.Series, pnl_series: pd.Series):
    fig = go.Figure()
    
    # Glowing equity line
    fig.add_trace(go.Scatter(x=equity.index, y=equity.values, 
                            mode="lines", name="EQUITY",
                            line=dict(color='#00ff88', width=4),
                            fill='tozeroy', fillcolor='rgba(0,255,136,0.1)'))
    
    # PnL bars with neon glow
    colors